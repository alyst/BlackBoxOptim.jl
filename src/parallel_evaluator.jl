typealias ChannelRef{T} @compat RemoteChannel{Channel{T}}

const PEStatus_Msg = 1
const PEStatus_OK = 0
const PEStatus_Stopped = -1
const PEStatus_Error = -2

"""
  Internal data for the worker process of the parallel evaluator.
"""
type ParallelEvaluatorWorker{T, P<:OptimizationProblem}
    id::Int
    problem::P
    param_status::SharedVector{Int}
    shared_param::SharedVector{T}
    fitness_status::SharedVector{Int}
    shared_fitness::SharedVector{T}

    @compat (::Type{ParallelEvaluatorWorker}){T, P<:OptimizationProblem}(
        id::Int, problem::P,
        param_status::SharedVector{Int}, shared_param::SharedVector{T},
        fitness_status::SharedVector{Int}, shared_fitness::SharedVector{T}) =
        new{T,P}(id, problem, param_status, shared_param, fitness_status, shared_fitness)
end

@inline param_status(worker::ParallelEvaluatorWorker) = worker.param_status[1]
@inline fitness_status(worker::ParallelEvaluatorWorker) = worker.fitness_status[1]

# run the wrapper (called in the "main" task)
function run!(worker::ParallelEvaluatorWorker)
    while true
        #info("Checking in worker #$(worker.id)")
        # continuously poll the worker for the delivery notification for
        # the last job or for the new job notification
        i = 0
        while param_status(worker) == PEStatus_OK || fitness_status(worker) == PEStatus_Msg
            if (i+=1) > 1000
                yield()
                i = 0
            end
        end
        # process the new worker status
        p_status = param_status(worker)
        if p_status == PEStatus_Stopped # master stopping
            worker.fitness_status[1] = PEStatus_Stopped # notify worker stopped
            break # shutdown!() called
        elseif p_status == PEStatus_Error # master error
            worker.fitness_status[1] = PEStatus_Stopped # stopped after receiving an error
            break
        elseif p_status == PEStatus_Msg # new job
            worker.param_status[1] = PEStatus_OK # received
            #info("PE worker #$(worker.id): got job")
            put_fitness!(worker.shared_fitness, fitness(worker.shared_param, worker.problem))
            worker.fitness_status[1] = PEStatus_Msg # fitness ready
        end
    end
end

"""
    Create and run the evaluator worker.
    The function that the master process spawns at each worker process.
"""
function run_parallel_evaluator_worker(id::Int,
                    worker_ready::ChannelRef{Int},
                    problem::OptimizationProblem,
                    param_status::SharedVector{Int},
                    shared_param::SharedVector{Float64},
                    fitness_status::SharedVector{Int},
                    shared_fitness::SharedVector{Float64})
  info("Initializing ParallelEvaluator worker #$id at task=$(myid())")
  worker = nothing
  try
    worker = ParallelEvaluatorWorker(id, deepcopy(problem),
                param_status, shared_param,
                fitness_status, shared_fitness)
  catch e
    # send -id to notify about an error and to release
    # the master from waiting for worker readiness
    warn("Exception at ParallelEvaluatorWorker initialization: $e")
    put!(worker_ready, -id)
    rethrow(e)
  end
  # create immigrants receiving tasks=#
  put!(worker_ready, id)
  info("Running worker #$id...")
  try
    run!(worker)
  catch e
    # send error candidate to notify about an error and to release
    # the master from waiting for worker messages
    warn("Exception while running ParallelEvaluatorWorker: $e")
    worker.fitness_status[1] = PEStatus_Error
    rethrow(e)
  end
  info("Worker #$id stopped")
  nothing
end

typealias PECandidateDict{FA} Dict{Int, Candidate{FA}}

"""
    Fitness evaluator that distributes fitness calculation
    among several worker processes.

    Currently the performance is limited
"""
type ParallelEvaluator{F, FA, T, FS, P<:OptimizationProblem, A<:Archive} <: Evaluator{P}
    problem::P
    archive::A
    num_evals::Int
    last_fitness::F
    arch_nafitness::FA

    params_status::Vector{SharedVector{Int}}
    shared_params::Vector{SharedVector{T}}
    fitnesses_status::Vector{SharedVector{Int}}
    shared_fitnesses::Vector{SharedVector{T}}

    fitness_slots::Base.Semaphore

    waiting_candidates::PECandidateDict{FA}
    unclaimed_candidates::PECandidateDict{FA} # done candidates that were not yet checked for completion

    worker2job::Vector{Int}

    job_assignment::ReentrantLock

    max_seq_done_job::Int   # all jobs from 1 to max_seq_done_job are done
    max_done_job::Int       # max Id of done job
    done_jobs::IntSet       # done job Ids beyond max_seq_done_job

    is_stopping::Bool
    next_job_id::Int

    worker_refs::Vector{ChannelRef{Any}}
    workers_handler::Task

    @compat (::Type{ParallelEvaluator}){P<:OptimizationProblem, A<:Archive}(
        problem::P, archive::A;
        pids::AbstractVector{Int} = workers())

        fs = fitness_scheme(problem)
        F = fitness_type(fs)
        T = fitness_eltype(fs)
        FA = fitness_type(archive)

        param_status = []

        etor = new{F, FA, T, typeof(fs), P, A}(
            problem, archive,
            0, nafitness(fs), nafitness(FA),
            [fill!(SharedArray(Int, (2,), pids=vcat(pid,[myid()])), 0) for pid in pids],
            [SharedArray(T, (numdims(problem),), pids=vcat(pid,[myid()])) for pid in pids],
            [fill!(SharedArray(Int, (2,), pids=vcat(pid,[myid()])), 0) for pid in pids],
            [SharedArray(T, (numobjectives(fs),), pids=vcat(pid,[myid()])) for pid in pids],
            Base.Semaphore(length(pids)),
            PECandidateDict{FA}(), PECandidateDict{FA}(),
            zeros(length(pids)), ReentrantLock(), 0, 0, IntSet(),
            false, 1
        )
        etor.worker_refs = _create_workers(etor, pids)
        etor.workers_handler = @schedule workers_handler!(etor)

        #finalizer(etor, _shutdown!)

        return etor
    end

    @compat (::Type{ParallelEvaluator})(
        problem::OptimizationProblem;
        pids::AbstractVector{Int} = workers(),
        archiveCapacity::Integer = 10) =
        ParallelEvaluator(problem, TopListArchive(fitness_scheme(problem), numdims(problem), archiveCapacity),
                          pids=pids)
end

nworkers(etor::ParallelEvaluator) = length(etor.worker_refs)
queue_capacity(etor::ParallelEvaluator) = nworkers(etor)

"""
    Count the candidates submitted (including the completed ones),
    but not yet claimed.
"""
queue_length(etor::ParallelEvaluator) = length(etor.waiting_candidates) + length(etor.unclaimed_candidates)

num_evals(etor::ParallelEvaluator) = etor.num_evals

is_stopping(etor::ParallelEvaluator) = etor.is_stopping

# check that worker is stil running.
# If running, its RemoteChannels should not be ready,
# but if there was exception in the worker,
# it would be thrown into the main thread
function check_worker_running{T}(worker::ChannelRef{T})
    if isready(worker)
        worker_res = fetch(worker) # fetch the worker, this should trigger an exception
        # no exception, but the worker should not be ready
        error("Worker at pid=$(worker.where) has finished before the master shutdown: $worker_res")
    end
    return true
end

function _create_workers(etor::ParallelEvaluator, pids::AbstractVector{Int})
    info("Initializing parallel workers...")
    workers_ready = @compat RemoteChannel(() -> Channel{Int}(length(pids))) # FIXME do we need to wait for the worker?

    # spawn workers
    problem = etor.problem
    params_status = etor.params_status
    shared_params = etor.shared_params
    fitnesses_status = etor.fitnesses_status
    shared_fitnesses = etor.shared_fitnesses

    worker_refs = ChannelRef{Any}[@spawnat(pid, run_parallel_evaluator_worker(i,
                       workers_ready, problem,
                       params_status[i], shared_params[i],
                       fitnesses_status[i], shared_fitnesses[i])) for (i, pid) in enumerate(pids)]
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
            error("Exception in the worker #$(-worker_id), but all workers still running")
        end
        info("  Worker #$worker_id is ready")
        nready += 1
    end
    info("All workers ready")
    return worker_refs
end

function shutdown!(etor::ParallelEvaluator)
    info("shutdown!(ParallelEvaluator)")
    !etor.is_stopping || error("Cannot shutdown!(ParallelEvaluator) twice")
    etor.is_stopping = true
    # notify the workers that they should shutdown (each worker should pick exactly one message)
    _shutdown!(etor)
    # resume workers handler if it is waiting for the new jobs
    lock(etor.job_assignment)
    unlock(etor.job_assignment)
    # wait for all the workers
    for i in 1:nworkers(etor)
        Base.acquire(etor.fitness_slots)
    end
    @assert !any(isposdef, etor.worker2job) "Some workers not finished"
    # release any waiting
    for i in 1:nworkers(etor)
        Base.release(etor.fitness_slots)
    end
end

function _shutdown!(etor::ParallelEvaluator)
    #info("_shutdown!(ParallelEvaluator)")
    if !etor.is_stopping
        etor.is_stopping = true
        #close(etor.in_fitnesses)
        #close(etor.out_individuals)
    end
    for i in 1:nworkers(etor)
        etor.params_status[i][1] = PEStatus_Stopped
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

function process_fitness{F}(etor::ParallelEvaluator{F}, worker_ix::Int)
    update_archive!(etor, job_id, new_fitness)
end

function update_archive!{F}(etor::ParallelEvaluator{F}, job_id::Int, fness::F)
    # update the list of done jobs
    #info("update_archive()")
    candi = pop!(etor.waiting_candidates, job_id)
    etor.unclaimed_candidates[job_id] = candi
    @assert length(etor.unclaimed_candidates) <= 1000_000 # sanity check
    update_done_jobs!(etor, job_id)
    etor.last_fitness = fness
    candi.fitness = archived_fitness(fness, etor.archive)
    etor.num_evals += 1
    #info("update_archive(): add_candidate()")
    add_candidate!(etor.archive, candi.fitness, candi.params, candi.tag, etor.num_evals)
    nothing
end

"""
    Process all incoming "fitness ready" messages until the evaluator is stopped.
"""
function workers_handler!{F}(etor::ParallelEvaluator{F})
    info("workers_handler!() started")
    while !is_stopping(etor) || !isempty(etor.waiting_candidates)
        # master critical section
        @inbounds for worker_ix in 1:nworkers(etor)
            #info("workers_handler!(): checking worker #$worker_ix...")
            #@assert check_worker_running(etor.worker_refs[worker_ix])
            if (job_id = etor.worker2job[worker_ix]) > 0 &&
               (fitness_status = etor.fitnesses_status[worker_ix][1]) != PEStatus_OK
                @assert (fitness_status == PEStatus_Msg || is_stopping(etor)) "Worker #$worker_ix bad status: $(fitness_status)"
                #info("worker_handler!(): fitness_evaluated")
                lock(etor.job_assignment)
                param_status = etor.params_status[worker_ix][1]
                new_fitness = get_fitness(F, etor.shared_fitnesses[worker_ix])
                @assert job_id > 0

                #info("worker_handler!($worker_ix): got fitness for job #$job_id")
                etor.worker2job[worker_ix] = 0 # clear job state

                etor.fitnesses_status[worker_ix][1] = PEStatus_OK # received
                unlock(etor.job_assignment)
                Base.release(etor.fitness_slots)

                if param_status == PEStatus_OK # communication in normal state, update the archive
                    update_archive!(etor, job_id, new_fitness)
                elseif param_status < 0 # error/stopping on the master side
                    # remove the candidate
                    delete!(etor.waiting_candidates, job_id)
                end
                #info("workers_handler!(): yield to other tasks after archive update")
                #yield() # free slots available, switch to the main task
            end
        end
        if length(etor.waiting_candidates) < nworkers(etor)
            if !is_stopping(etor) && isempty(etor.waiting_candidates)
                wait(etor.job_assignment.cond_wait)
            else
                #info("workers_handler!(): yield to other tasks")
                if !isempty(fitness_done(etor).waitq)
                    # somebody still waiting, notify
                    notify(fitness_done(etor))
                end
                yield() # free slots available, switch to the main task
            end
        end
    end
    info("workers_handler!() stopped")
end

"""
    Asynchronously calculate the fitness of a candidate.
    If `force`, existing fitness would be re-evaluated.

    Returns -1 if fitness is already evaluated,
            0 if no fitness evaluation was scheduled (`wait=false` and all workers occupied),
            id of fitness evaluation job (check status using `isready()`)
"""
function async_update_fitness{F,FA}(etor::ParallelEvaluator{F,FA}, candi::Candidate{FA}; force::Bool=false, wait::Bool=false)
    #info("async_update_fitness(): starting to assign job #$(etor.next_job_id)")
    if !etor.is_stopping && (force || isnafitness(fitness(candi), fitness_scheme(etor.archive)))
        if length(etor.waiting_candidates) >= queue_capacity(etor) && !wait
            #info("async_update_fitness(): queue is full, skip")
            return 0 # queue full, job not submitted
        end
        #info("async_update_fitness(): waiting to assign job #$(etor.next_job_id)")
        Base.acquire(etor.fitness_slots)
        #info("async_update_fitness(): initial slot_state: $(etor.worker2job), $(etor.fitness_slots.curr_cnt)")
        lock(etor.job_assignment)
        worker_ix = findfirst(etor.worker2job, 0)
        @assert (worker_ix > 0) "Cannot find a worker #$(worker_ix) to put a job to"
        etor.worker2job[worker_ix] = job_id = etor.next_job_id
        etor.next_job_id += 1
        copy!(etor.shared_params[worker_ix], candi.params) # share candidate with the workers
        #info("async_update_fitness(): assigning job #$job_id to worker #$worker_ix")
        etor.waiting_candidates[job_id] = candi
        #info("async_update_fitness(): assert fitness status")
        #@assert etor.fitnesses_status[worker_ix][1] == PEStatus_OK
        #@assert etor.params_status[worker_ix][1] == PEStatus_OK
        #info("async_update_fitness(): flip param status")
        etor.params_status[worker_ix][1] = PEStatus_Msg # ready
        #info("async_update_fitness(): assigned job #$job_id to worker #$worker_ix")
        #info("async_update_fitness(): unlock job assignment")
        unlock(etor.job_assignment)
        #info("async_update_fitness(): yield()")
        #yield() # dispatch the job ASAP, without this it's not getting queued
        return job_id
    else
        return -1 # the candidate has fitness, skip recalculation
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
         Candidate{FA}(Individual(), -1, etor.arch_nafitness)) # job was claimed
    return fit_job_id <= etor.max_seq_done_job || in(fit_job_id, etor.done_jobs)
end

"""
    Processes all completed but not yet claimed candidates.
    `f` accepts the completed fitness job Id and corresponding candidate,
    returns `true` if the candidate was successfully claimed.
"""
function process_completed!(f::Function, etor::ParallelEvaluator)
    for (job_id, candi) in etor.unclaimed_candidates
        if f(job_id, candi)
            # remove job_id from the waiting list and from the unclaimed list
            #info("process_completed!($job_id)")
            delete!(etor.unclaimed_candidates, job_id)
        end
    end
    return etor
end

"""
    fitness_done(etor::ParallelEvaluator)

    Get the condition that is triggered each time fitness evaluation completes.
"""
@inline fitness_done(etor::ParallelEvaluator) = etor.fitness_slots.cond_wait

"""
    Calculate fitness for given candidates.
    Waits until all fitnesses have been calculated.
"""
function update_fitness!{F,FA}(etor::ParallelEvaluator{F,FA}, candidates::Vector{Candidate{FA}}; force::Bool=false)
    job_ids = sizehint!(IntSet(), length(candidates))
    n_pending = 0
    for candi in candidates
        job_id = async_update_fitness(etor, candi, force=force, wait=true)
        #info("update_fitness!(): got job id #$job_id")
        if job_id > 0
            n_pending += 1
            push!(job_ids, job_id)
        end
    end
    # wait until it's done
    while !is_stopping(etor) && n_pending > 0 &&
          !(isempty(etor.waiting_candidates) && isempty(etor.unclaimed_candidates))
        #info("job_ids: $job_ids")
        process_completed!(etor) do job_id, candi
            our_job = pop!(job_ids, job_id, 0)>0
            if our_job
                n_pending-=1
            end
            return our_job
        end
        if n_pending > 0 && isempty(etor.unclaimed_candidates)
            #info("update_fitness!(): wait()")
            wait(fitness_done(etor))
        end
    end
    @assert (n_pending == 0) "Fitnesses not evaluated (#$job_ids)"
    return candidates
end

# FIXME it's not efficient to calculate fitness like that with `ParallelEvaluator`
function fitness{F,FA}(params::Individual, etor::ParallelEvaluator{F,FA})
    candi = Candidate{FA}(params, -1, etor.arch_nafitness)
    job_id = async_update_fitness(etor, candi, wait=true)
    while !is_stopping(etor) &&
          !(isempty(etor.waiting_candidates) && isempty(etor.unclaimed_candidates))
        if isready(etor, job_id)
            #info("fitness(): done")
            return fitness(candi)
        else
            #info("fitness(): wait()")
            wait(fitness_done(etor))
        end
    end
    error("Fitness not evaluated")
end
