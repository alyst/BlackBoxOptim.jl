typealias PESlotChannel Channel{Int} # channel for exchaning slot Ids
typealias PESlotChannelRef RemoteRef{PESlotChannel}

"""
  Internal data for the worker process of the parallel evaluator.
"""
immutable ParallelEvaluatorWorker{P<:OptimizationProblem}
    id::Int
    problem::P
    param_slots::SharedMatrix{Float64}
    fitness_slots::SharedMatrix{Float64}
    in_params::PESlotChannelRef
    out_fitnesses::PESlotChannelRef

    Base.call{P<:OptimizationProblem}(::Type{ParallelEvaluatorWorker}, id::Int, problem::P,
        param_slots::SharedMatrix{Float64}, fitness_slots::SharedMatrix{Float64},
        in_params::PESlotChannelRef, out_fitnesses::PESlotChannelRef) =
        new{P}(id, problem, param_slots, fitness_slots, in_params, out_fitnesses)
end

# run the wrapper (called in the "main" task)
function run!(worker::ParallelEvaluatorWorker)
    params = Individual(numdims(worker.problem))
    while (slot_id = take!(worker.in_params)) > 0
        #info("PE worker #$(worker.id): got job in slot #$(slot_id)")
        copy!(params, slice(worker.param_slots, :, slot_id))
        put_fitness!(worker.fitness_slots, fitness(params, worker.problem), slot_id)
        put!(worker.out_fitnesses, slot_id)
    end
end

# Function that the master process spawns at each worker process.
# Creates and run the worker wrapper
function run_parallel_evaluator_worker(id::Int,
                    worker_ready::RemoteRef{Channel{Int}},
                    problem::OptimizationProblem,
                    param_slots::SharedMatrix{Float64},
                    fitness_slots::SharedMatrix{Float64},
                    in_params::PESlotChannelRef,
                    out_fitnesses::PESlotChannelRef)
  info("Initializing ParallelEvaluator worker #$id at task=$(myid())")
  worker = nothing
  try
    worker = ParallelEvaluatorWorker(id, problem,
                param_slots, fitness_slots,
                in_params, out_fitnesses)
  catch e
    # send -id to notify about an error and to release
    # the master from waiting for worker readiness
    info("Exception: $e")
    put!(worker_ready, -id)
    rethrow(e)
  end
  # create immigrants receiving tasks=#
  put!(worker_ready, id)
  info("Running worker #$id")
  try
    run!(worker)
  catch e
    # send error candidate to notify about an error and to release
    # the master from waiting for worker messages
    info("Exception: $e")
    put!(worker.out_fitnesses, -1)
    rethrow(e)
  end
  info("Worker #$id stopped")
  nothing
end

typealias PECandidateDict{FA} Dict{Int, Candidate{FA}}

"""
    Fitness evaluator that distributes candidates fitness calculation
    among several worker processes.
"""
type ParallelEvaluator{F, FA, FS, P<:OptimizationProblem, A<:Archive} <: Evaluator{P}
    problem::P
    archive::A
    num_evals::Int
    last_fitness::F

    param_slots::SharedMatrix{Float64}
    fitness_slots::SharedMatrix{Float64}
    out_params::Vector{PESlotChannelRef}
    in_fitnesses::PESlotChannelRef

    waiting_candidates::PECandidateDict{FA}
    unclaimed_candidates::PECandidateDict{FA} # done candidates that were not yet checked for completion

    slot2job::Vector{Int}
    slot2worker::Vector{Int}
    worker_njobs::Vector{Int}
    njobs2workers::Vector{Vector{Int}}

    max_seq_done_job::Int   # all jobs from 1 to max_seq_done_job are done
    max_done_job::Int       # max Id of done job
    done_jobs::IntSet       # done job Ids beyond max_seq_done_job

    fitness_evaluated::Condition

    is_stopping::Bool
    next_job_id::Int

    recv_task::Task
    worker_refs::Vector{RemoteRef{Channel{Any}}}

    function Base.call{P<:OptimizationProblem, A<:Archive}(::Type{ParallelEvaluator},
        problem::P, archive::A;
        pids::AbstractVector{Int} = workers(),
        queueCapacity::Integer = length(pids))

        fs = fitness_scheme(problem)
        F = fitness_type(fs)
        FA = archived_fitness_type(archive)

        paramChanCapacity = queueCapacity÷length(pids)+1

        etor = new{F, FA, typeof(fs), P, A}(
            problem, archive,
            0, nafitness(fs),
            SharedArray(Float64, (numdims(problem), queueCapacity), pids=vcat(pids,[myid()])),
            SharedArray(Float64, (numobjectives(fs), queueCapacity), pids=vcat(pids,[myid()])),
            PESlotChannelRef[RemoteRef(() -> PESlotChannel(paramChanCapacity), pid) for pid in pids],
            RemoteRef(() -> PESlotChannel(queueCapacity), myid()),
            PECandidateDict{FA}(), PECandidateDict{FA}(),
            zeros(queueCapacity), zeros(queueCapacity),
            zeros(length(pids)), Vector{Int}[i == 0 ? collect(1:length(pids)) : Vector{Int}() for i in 0:queueCapacity],
            0, 0, IntSet(),
            Condition(), false, 1
        )

        # "background" fitnesses receiver task
        etor.recv_task = @schedule recv_fitnesses!(etor)

        #finalizer(etor, _shutdown!)

        etor.worker_refs = _create_workers(etor, pids)

        return etor
    end

    Base.call(::Type{ParallelEvaluator},
        problem::OptimizationProblem;
        pids::AbstractVector{Int} = workers(),
        queueCapacity::Integer = length(pids),
        archiveCapacity::Integer = 10) =
        ParallelEvaluator(problem, TopListArchive(fitness_scheme(problem), numdims(problem), archiveCapacity),
                          pids=pids, queueCapacity=queueCapacity)
end

nworkers(etor::ParallelEvaluator) = length(etor.worker_refs)
queue_capacity(etor::ParallelEvaluator) = length(etor.slot2job)

"""
    Count the candidates submitted (including the completed ones),
    but not yet claimed.
"""
queue_length(etor::ParallelEvaluator) = length(etor.waiting_candidates) + length(etor.unclaimed_candidates)

num_evals(etor::ParallelEvaluator) = etor.num_evals

is_stopping(etor::ParallelEvaluator) = etor.is_stopping

# check that worker is stil running.
# If running, its RemoteRefs should not be ready,
# but if there was exception in the worker,
# it would be thrown into the main thread
function check_worker_running{T}(worker::RemoteRef{T})
    if isready(worker)
        fetch(worker) # fetch the worker, this should trigger an exception
        # no exception, but the worker should not be ready
        error("Worker #? has finished before the master shutdown")
    end
    return true
end

function _create_workers(etor::ParallelEvaluator, pids::AbstractVector{Int})
    info("Initializing parallel workers...")
    workers_ready = RemoteRef(() -> Channel{Int}(length(pids))) # FIXME do we need to wait for the worker?

    # spawn workers
    problem = etor.problem
    param_slots = etor.param_slots
    fitness_slots = etor.fitness_slots
    out_params = etor.out_params
    in_fits = etor.in_fitnesses
    worker_refs = RemoteRef{Channel{Any}}[@spawnat(pid, run_parallel_evaluator_worker(i,
                       workers_ready, problem, param_slots, fitness_slots,
                       out_params[i], in_fits)) for (i, pid) in enumerate(pids)]
    #@assert !isready(ppopt.is_started)
    # wait until all the workers are started
    info("Waiting for the workers to be ready...")
    # FIXME is it required?
    nready = 0
    while nready < length(pids)
        map(check_worker_running, worker_refs)
        worker_id = take!(workers_ready)
        if worker_id < 0
            # worker failed to initialize, reading its task would throw an exception
            check_worker_running(worker_refs[-worker_id])
            error("Exception in the worker, but all workers still running")
        end
        info("  Worker #$worker_id is ready")
        nready += 1
    end
    info("All workers ready")
    return worker_refs
end

function shutdown!(etor::ParallelEvaluator)
    info("shutdown!(ParallelEvaluator)")
    if etor.is_stopping
        throw(InternalError("Cannot shutdown!(ParallelEvaluator) twice"))
    end
    etor.is_stopping = true
    # notify the workers that they should shutdown (each worker should pick exactly one message)
    for i in 1:nworkers(etor)
        put!(etor.out_params[i], -1)
        close(etor.out_params[i])
    end
    close(etor.in_fitnesses)
    notify(etor.fitness_evaluated)
    _shutdown!(etor)
end

function _shutdown!(etor::ParallelEvaluator)
    info("_shutdown!(ParallelEvaluator)")
    if !etor.is_stopping
        etor.is_stopping = true
        #close(etor.in_fitnesses)
        #close(etor.out_individuals)
    end
    etor
end

function update_done_jobs!(etor::ParallelEvaluator, job_id)
    if job_id > etor.max_done_job
        etor.max_done_job = job_id
    end
    if job_id == etor.max_seq_done_job+1
        # the next sequential job
        etor.max_seq_done_job = job_id
        # see if max_seq_done_job could be further advanced using done jobs
        while etor.max_done_job > etor.max_seq_done_job && first(etor.done_jobs) == etor.max_seq_done_job+1
            etor.max_seq_done_job += 1
            shift!(etor.done_jobs)
        end
    else
        push!(etor.done_jobs, job_id)
    end
end

"""
    Process all incoming "fitness ready" messages until the evaluator is stopped.
"""
function recv_fitnesses!{F}(etor::ParallelEvaluator{F})
    #info("recv_fitnesses()")
    while !is_stopping(etor) && (slot_id=take!(etor.in_fitnesses))>0
        job_id = etor.slot2job[slot_id]
        @assert job_id > 0
        worker_id = etor.slot2worker[slot_id]
        @assert worker_id > 0
        worker_njobs = etor.worker_njobs[worker_id]
        @assert worker_njobs > 0
        # update worker pending jobs count
        etor.worker_njobs[worker_id] -= 1
        # update list of workers with given jobs count
        worker_ix = findfirst(etor.njobs2workers[worker_njobs+1], worker_id)
        @assert worker_ix > 0
        deleteat!(etor.njobs2workers[worker_njobs+1], worker_ix)
        push!(etor.njobs2workers[worker_njobs], worker_id)

        #info("recv_fitnesses!(): got fitness from worker #$worker_id for slot #$(slot_id), job #$job_id")
        candi = pop!(etor.waiting_candidates, job_id)
        etor.last_fitness = candi.fitness = archived_fitness(get_fitness(F, etor.fitness_slots, slot_id), etor.archive)
        etor.slot2job[slot_id] = 0 # clear job state
        etor.slot2worker[slot_id] = 0 # clear worker state
        etor.num_evals += 1
        add_candidate!(etor.archive, candi.fitness, candi.params, candi.tag, etor.num_evals)
        # update the list of done jobs
        etor.unclaimed_candidates[job_id] = candi
        if length(etor.unclaimed_candidates) > 1000_000 # sanity check
            throw(InternalError("Too many unclaimed candidates with evaluated fitness"))
        end
        update_done_jobs!(etor, job_id)
        #info("recv_fitnesses(): fitness_evaluated")
        notify(etor.fitness_evaluated)
    end
end

"""
    Asynchronously update fitness of a candidate.
    If `force`, existing fitness would be re-evaluated.

    Returns -1 if fitness is already evaluated,
            0 if no fitness evaluation was scheduled (job queue full),
            id of fitness evaluation job (check status using `isready()`)
"""
function async_update_fitness{F,FA}(etor::ParallelEvaluator{F,FA}, candi::Candidate{FA}; force::Bool=false, wait::Bool=false)
    if force || isnafitness(fitness(candi), fitness_scheme(etor.archive))
        # FIXME is Base.length(RemoteChannel) is available, use it
        if length(etor.waiting_candidates) >= queue_capacity(etor)
            # queue is full, refuse to submit another job
            if wait # FIXME race condition when slot freed before the wait for fitness evaluation started
                Base.wait(etor.fitness_evaluated)
            else
                return 0
            end
        end
        #info("async_update_fitness(): initial slot_state: $(etor.slot2job)")
        free_slot_id = findfirst(etor.slot2job, 0)
        @assert free_slot_id > 0
        # find a most free worker to assign a job
        worker_id = 0
        worker_njobs = 0
        for (njobs, workers) in enumerate(etor.njobs2workers)
            if !isempty(workers)
                worker_id = pop!(workers)
                worker_njobs = njobs-1
                break
            end
        end
        if worker_id == 0
            error("Cannot find a worker to put a job to")
        end
        @assert etor.worker_njobs[worker_id] == worker_njobs
        etor.slot2job[free_slot_id] = job_id = etor.next_job_id
        etor.slot2worker[free_slot_id] = worker_id
        push!(etor.njobs2workers[worker_njobs+2], worker_id)
        etor.worker_njobs[worker_id] += 1
        etor.next_job_id += 1
        etor.param_slots[:, free_slot_id] = candi.params # share candidate with the workers
        #info("async_update_fitness(): assigning job #$job_id to slot #$free_slot_id to worker #$worker_id")
        #info("async_update_fitness(): slot_state: $(etor.slot2job)")
        etor.waiting_candidates[job_id] = candi
        put!(etor.out_params[worker_id], free_slot_id)
        return job_id
    else
        return -1
    end
end

"""
    isready(etor::ParallelEvaluator, fit_job_id::Int)

    Check if given asynchronous fitness job calculation is complete.
    `fit_job_id` is assigned by `async_update_fitness()`.
"""
function Base.isready{F,FA}(etor::ParallelEvaluator{F,FA}, fit_job_id::Int)
    fit_job_id > 0 || throw(ArgumentError("Incompatible fitness job Id"))
    pop!(etor.unclaimed_candidates, fit_job_id,
         Candidate{FA}(Individual(), -1, nafitness(fitness_scheme(etor.archive)))) # job was claimed
    return fit_job_id <= etor.max_seq_done_job || in(fit_job_id, etor.done_jobs)
end

"""
    Processes all completed but not yet claimed candidates.
    `f` accepts the completed fitness job Id and corresponding candidate,
    returns `true` if the candidate was successfully claimed.
"""
function process_completed!(f::Function, etor::ParallelEvaluator)
    for (fit_id, candi) in etor.unclaimed_candidates
        if f(fit_id, candi)
            # remove fit_id from the waiting list and from the unclaimed list
            #info("process_completed!($fit_id)")
            delete!(etor.unclaimed_candidates, fit_id)
        end
    end
    return etor
end

"""
    Calculate fitness for given candidates.
    Waits until all fitnesses have been calculated.
"""
function update_fitness!{F,FA}(etor::ParallelEvaluator{F,FA}, candidates::Vector{Candidate{FA}}; force::Bool=false)
    fit_ids = sizehint!(IntSet(), length(candidates))
    for candi in candidates
        fit_id = async_update_fitness(etor, candi, force=force, wait=true)
        if fit_id > 0
            push!(fit_ids, fit_id)
        end
    end
    # wait until it's done
    while !is_stopping(etor) && !isempty(fit_ids) &&
          !(isempty(etor.waiting_candidates) && isempty(etor.unclaimed_candidates))
        #info("fit_ids: $fit_ids")
        process_completed!((fit_id, candi) -> pop!(fit_ids, fit_id, 0)>0, etor)
        if !isempty(fit_ids) && isempty(etor.unclaimed_candidates)
            #info("update_fitness!(): wait(fitness_evaluated)")
            wait(etor.fitness_evaluated)
        end
    end
    if !isempty(fit_ids)
        throw(InternalError("Fitnesses not evaluated ($fit_ids)"))
    end
    return candidates
end

"""
    wait(etor::ParallelEvaluator)

    Wait until any queued fitness evaluation is complete.
"""
Base.wait(etor::ParallelEvaluator) = wait(etor.fitness_evaluated)

# FIXME it's not efficient to calculate fitness like that with `ParallelEvaluator`
function fitness{F,FA}(params::Individual, etor::ParallelEvaluator{F,FA})
    candi = Candidate{FA}(params)
    id = async_update_fitness(etor, candi)
    while !etor.is_stopping &&
          !(isempty(etor.waiting_candidates) && isempty(etor.unclaimed_candidates))
        #info("fitness(): wait(fitness_evaluated)")
        wait(etor.fitness_evaluated)
        if isready(etor, id)
            return fitness(candi)
        end
    end
    throw(InternalError("Fitness not evaluated"))
end
