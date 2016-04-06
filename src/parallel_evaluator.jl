const PE_FINAL_CANDIDATE = -12345 # special terminating candidate with worker index sent by master
const PE_ERROR_CANDIDATE = -67890 # special error candidate with worker index send by worker

"""
    `ParallelEvaluator` message with the candidate passed from the master to the worker.
"""
immutable PEInputMessage
  id::Int                   # candidate id
  indi::Individual
end

"""
    `ParallelEvaluator` output message with the fitness calculated by the worker.
"""
immutable PEOutputMessage{F}
  id::Int                   # candidate id
  fitness::F
end

typealias PEInputChannel Channel{PEInputMessage}
typealias PEInputChannelRef RemoteRef{PEInputChannel}

typealias PEOutputChannel{F} Channel{PEOutputMessage{F}}
typealias PEOutputChannelRef{F} RemoteRef{PEOutputChannel{F}}

"""
  Internal data for the worker process of the parallel evaluator.
"""
immutable ParallelEvaluatorWorker{F, P<:OptimizationProblem}
    id::Int
    problem::P
    in_individuals::PEInputChannelRef
    out_fitnesses::PEOutputChannelRef{F}

    Base.call{F, P<:OptimizationProblem}(::Type{ParallelEvaluatorWorker}, id::Int, problem::P,
        in_individuals::PEInputChannelRef, out_fitnesses::PEOutputChannelRef{F}) =
        new{F, P}(id, problem, in_individuals, out_fitnesses)
end

# run the wrapper (called in the "main" task)
function run!{F}(worker::ParallelEvaluatorWorker{F})
    while (indi_msg = take!(worker.in_individuals)).id != PE_FINAL_CANDIDATE
        #info("PE worker #$(worker.id): got job id #$(indi_msg.id)")
        put!(worker.out_fitnesses, PEOutputMessage{F}(indi_msg.id, fitness(indi_msg.indi, worker.problem)))
    end
end

# Function that the master process spawns at each worker process.
# Creates and run the worker wrapper
function run_parallel_evaluator_worker{F}(id::Int,
                    worker_ready::RemoteRef{Channel{Int}},
                    problem::OptimizationProblem,
                    in_individuals::PEInputChannelRef,
                    out_fitnesses::PEOutputChannelRef{F})
  info("Initializing ParallelEvaluator worker #$id at task=$(myid())")
  worker = nothing
  try
    worker = ParallelEvaluatorWorker(id, problem,
                in_individuals, out_fitnesses)
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
    put!(worker.out_fitnesses,
         PEOutputMessage{F}(ERROR_CANDIDATE, nafitness(fitness_scheme(problem))))
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

    queue_capacity::Int     # maximum length of out_individuals channel (#FIXME if capacity(RemoteChannel) is available, use it)
    out_individuals::PEInputChannelRef
    in_fitnesses::PEOutputChannelRef{F}

    waiting_candidates::PECandidateDict{FA}
    unclaimed_candidates::PECandidateDict{FA} # done candidates that were not yet checked for completion

    max_seq_done_job::Int   # all jobs from 1 to max_seq_done_job are done
    max_done_job::Int       # max Id of done job
    done_jobs::IntSet       # done job Ids beyond max_seq_done_job

    fitness_evaluated::Condition

    is_stopping::Bool
    next_id::Int

    recv_task::Task
    worker_refs::Vector{RemoteRef{Channel{Any}}}

    function Base.call{P<:OptimizationProblem, A<:Archive}(::Type{ParallelEvaluator},
        problem::P, archive::A;
        pids::AbstractVector{Int} = workers(),
        queueCapacity::Integer = length(pids))

        fs = fitness_scheme(problem)
        F = fitness_type(fs)
        FA = archived_fitness_type(archive)

        out_individuals = RemoteRef(() -> PEInputChannel(queueCapacity), myid())
        in_fitnesses = RemoteRef(() -> PEOutputChannel{F}(5*queueCapacity), myid())

        etor = new{F, FA, typeof(fs), P, A}(
            problem, archive,
            0, nafitness(fs),
            queueCapacity, out_individuals, in_fitnesses,
            PECandidateDict{FA}(), PECandidateDict{FA}(),
            0, 0, IntSet(),
            Condition(), false, 1
        )

        # "background" fitnesses receiver task
        etor.recv_task = @schedule while !etor.is_stopping
            #info("recv_task: wait(in_fitnesses)")
            wait(etor.in_fitnesses)
            recv_fitnesses!(etor)
        end

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
queue_capacity(etor::ParallelEvaluator) = etor.queue_capacity

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
    out_indis = etor.out_individuals
    in_fits = etor.in_fitnesses
    worker_refs = RemoteRef{Channel{Any}}[@spawnat(pid, run_parallel_evaluator_worker(i,
                       workers_ready, problem, out_indis, in_fits)) for (i, pid) in enumerate(pids)]
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
        put!(etor.out_individuals, PEInputMessage(PE_FINAL_CANDIDATE, Individual()))
    end
    close(etor.in_fitnesses)
    close(etor.out_individuals)
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

"""
    Process all the fitnesses currently in the `in_fitnesses` channel.
"""
function recv_fitnesses!{F}(etor::ParallelEvaluator{F})
    #info("recv_fitnesses()")
    while !etor.is_stopping && !isempty(etor.waiting_candidates) && isready(etor.in_fitnesses)
        msg = take!(etor.in_fitnesses)::PEOutputMessage{F}
        candi = pop!(etor.waiting_candidates, msg.id)
        etor.last_fitness = candi.fitness = archived_fitness(msg.fitness, etor.archive)
        etor.num_evals += 1
        add_candidate!(etor.archive, candi.fitness, candi.params, candi.tag, etor.num_evals)
        # update the list of done jobs
        etor.unclaimed_candidates[msg.id] = candi
        if length(etor.unclaimed_candidates) > 1000_000 # sanity check
            throw(InternalError("Too many unclaimed candidates with evaluated fitness"))
        end
        if msg.id > etor.max_done_job
            etor.max_done_job = msg.id
        end
        if msg.id == etor.max_seq_done_job + 1
            # the next sequential job
            etor.max_seq_done_job = msg.id
        else
            push!(etor.done_jobs, msg.id)
        end
        if !isempty(etor.done_jobs) && length(etor.done_jobs) == etor.max_done_job - etor.max_seq_done_job
            # empty IntSet of done jobs because all 1,2,...max_done_job jobs are done
            etor.max_seq_done_job = etor.max_done_job
            empty!(etor.done_jobs)
        end
    end
    notify(etor.fitness_evaluated)
end

"""
    Asynchronously update fitness of a candidate.
    If `force`, existing fitness would be re-evaluated.

    Returns 0 if fitness is already evaluated,
            -1 if no fitness evaluation was scheduled (job queue full),
            id of fitness evaluation job (check status using `isready()`)
"""
function async_update_fitness{F,FA}(etor::ParallelEvaluator{F,FA}, candi::Candidate{FA}; force::Bool=false, wait::Bool=false)
    if force || isnafitness(fitness(candi), fitness_scheme(etor.archive))
        # FIXME is Base.length(RemoteChannel) is available, use it
        if !wait && length(etor.waiting_candidates) >= etor.queue_capacity
            # queue is full, refuse to submit another job
            return -1
        end
        id = etor.next_id
        etor.next_id += 1
        #info("async_update_fitness: new fitness task #$id")
        etor.waiting_candidates[id] = candi
        put!(etor.out_individuals, PEInputMessage(id, candi.params))
        return id
    else
        return 0
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
            #info("Done $fit_id")
            delete!(etor.unclaimed_candidates, fit_id)
        end
    end
    return etor
end

"""
    Calculate fitness for given candidates.
    Waits until all fitness have been calculated.
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
    while !etor.is_stopping && !isempty(fit_ids) && !(isempty(etor.waiting_candidates) && isempty(etor.unclaimed_candidates))
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
    while !etor.is_stopping
        #info("fitness(): wait(fitness_evaluated)")
        wait(etor.fitness_evaluated)
        if isready(etor, id)
            return fitness(candi)
        end
    end
    throw(InternalError("Fitness not evaluated"))
end
