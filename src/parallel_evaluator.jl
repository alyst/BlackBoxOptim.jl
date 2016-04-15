typealias PESlotChannel Channel{Int} # channel for exchaning slot Ids
typealias PESlotChannelRef RemoteRef{PESlotChannel}

"""
  Internal data for the worker process of the parallel evaluator.
"""
type ParallelEvaluatorWorker{T, P<:OptimizationProblem}
    id::Int
    problem::P
    shared_param::SharedVector{T}
    shared_fitness::SharedVector{T}

    Base.call{T, P<:OptimizationProblem}(::Type{ParallelEvaluatorWorker}, id::Int, problem::P,
        shared_param::SharedVector{T}, shared_fitness::SharedVector{T}) =
        new{T,P}(id, problem, shared_param, shared_fitness)
end

typealias ParallelEvaluatorWorkerRef{T,P} RemoteRef{Channel{ParallelEvaluatorWorker{T,P}}}

@inline function Base.call(worker::ParallelEvaluatorWorker)
    put_fitness!(worker.shared_fitness, fitness(worker.shared_param, worker.problem))
    return nothing
end

@inline Base.call{T,P}(worker_ref::ParallelEvaluatorWorkerRef{T,P}) =
    (fetch(worker_ref)::ParallelEvaluatorWorker{T,P})()

typealias PECandidateDict{FA} Dict{Int, Candidate{FA}}

# HACK to avoid serializing the same remotecall message every time
# First, the message gets cached
# Then the same binary blob is being send by `remotecall_fetch_cached`
# Since remotecall status Remote Ref Id is a part of the message, it also gets reused.
function remotecall_fetch_msg(w::Base.Worker, f, args...)
    # can be weak, because the program will have no way to refer to the Ref
    # itself, it only gets the result.
    oid = Base.next_rrid_tuple()
    rv = Base.lookup_ref(oid)
    rv.waitingfor = w.id
    Base.CallMsg{:call_fetch}(f, args, oid), oid
end

function cache_msg(msg)
    cache=IOBuffer()
    serialize(cache, msg)
    takebuf_array(cache)
end

function send_cached_msg(w::Base.Worker, cached_msg::Vector{UInt8}, now::Bool)
    Base.check_worker_state(w)
    io = w.w_stream
    lock(io.lock)
    try
        write(io, cached_msg)

        if !now && w.gcflag
            Base.flush_gc_msgs(w)
        else
            flush(io)
        end
    finally
        unlock(io.lock)
    end
end

function remotecall_fetch_cached(w::Base.Worker, cached_msg, cached_oid)
    # can be weak, because the program will have no way to refer to the Ref
    # itself, it only gets the result.
    rv = Base.lookup_ref(cached_oid)
    rv.waitingfor = w.id
    send_cached_msg(w, cached_msg, true)
    v = take!(rv)
    delete!(Base.PGRP.refs, cached_oid)
    isa(v, RemoteException) ? throw(v) : v
end

"""
    Fitness evaluator that distributes fitness calculation
    among several worker processes.
"""
type ParallelEvaluator{F, FA, T, FS, P<:OptimizationProblem, A<:Archive} <: Evaluator{P}
    problem::P
    archive::A
    num_evals::Int
    last_fitness::F

    shared_params::Vector{SharedVector{T}}
    shared_fitnesses::Vector{SharedVector{T}}

    worker_onhold::Vector{Base.Semaphore}
    worker_refs::Vector{ParallelEvaluatorWorkerRef{T,P}}
    worker_handlers::Vector{Task}

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

    function Base.call{P<:OptimizationProblem, A<:Archive}(::Type{ParallelEvaluator},
        problem::P, archive::A;
        pids::AbstractVector{Int} = workers())

        fs = fitness_scheme(problem)
        F = fitness_type(fs)
        T = fitness_eltype(fs)
        FA = archived_fitness_type(archive)

        shared_params = [SharedArray(T, (numdims(problem),), pids=vcat(pid,[myid()])) for pid in pids]
        shared_fitnesses = [SharedArray(T, (numobjectives(fs),), pids=vcat(pid,[myid()])) for pid in pids]
        worker_refs = [RemoteRef(function ()
             # create fake channel and put problem there
             ch = Channel{ParallelEvaluatorWorker{T,P}}(1)
             put!(ch, ParallelEvaluatorWorker(i, copy(problem),
                shared_params[i], shared_fitnesses[i]))
             ch
        end, pid) for (i, pid) in enumerate(pids)]

        etor = new{F, FA, T, typeof(fs), P, A}(
            problem, archive,
            0, nafitness(fs),
            shared_params, shared_fitnesses,
            [Base.Semaphore(0) for _ in pids],
            worker_refs,
            [@schedule worker_handler!(etor, Base.worker_from_id(pid), i) for (i, pid) in enumerate(pids)],
            Base.Semaphore(length(pids)),
            PECandidateDict{FA}(), PECandidateDict{FA}(),
            zeros(length(pids)), ReentrantLock(), 0, 0, IntSet(),
            false, 1
        )

        #finalizer(etor, _shutdown!)

        return etor
    end

    Base.call(::Type{ParallelEvaluator},
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
    notify(etor.fitness_slots.cond_wait) # release any waiting tasks
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

function process_fitness{F}(etor::ParallelEvaluator{F}, worker_ix::Int)
    lock(etor.job_assignment)
    job_id = etor.worker2job[worker_ix]
    new_fitness = get_fitness(F, etor.shared_fitnesses[worker_ix])
    @assert job_id > 0

    #info("worker_handler!($worker_ix): got fitness for job #$job_id")
    candi = pop!(etor.waiting_candidates, job_id)
    etor.worker2job[worker_ix] = 0 # clear job state
    etor.num_evals += 1
    # update the list of done jobs
    etor.unclaimed_candidates[job_id] = candi
    if length(etor.unclaimed_candidates) > 1000_000 # sanity check
        throw(InternalError("Too many unclaimed candidates with evaluated fitness"))
    end
    update_done_jobs!(etor, job_id)
    unlock(etor.job_assignment)
    Base.release(etor.fitness_slots)

    etor.last_fitness = candi.fitness = archived_fitness(new_fitness, etor.archive)
    add_candidate!(etor.archive, candi.fitness, candi.params, candi.tag, etor.num_evals)
end

"""
    Process all incoming "fitness ready" messages until the evaluator is stopped.
"""
function worker_handler!(etor::ParallelEvaluator, proc::Base.Worker, worker_ix::Int)
    #info("worker_handler($worker_ix): starting")
    worker_ref = etor.worker_refs[worker_ix]
    ping_msg, ping_oid = remotecall_fetch_msg(proc, Base.call, worker_ref)
    cached_ping_msg = cache_msg(ping_msg)

    while !is_stopping(etor)
        #info("worker_handler!($worker_ix): waiting....")
        Base.acquire(etor.worker_onhold[worker_ix])
        remotecall_fetch_cached(proc, cached_ping_msg, ping_oid)
        process_fitness(etor, worker_ix)
        #info("worker_handler!($worker_ix): notify job #$job_id done....")
    end
end

function worker_handler!(etor::ParallelEvaluator, proc::LocalProcess, worker_ix::Int)
    #info("worker_handler($worker_ix): starting")
    worker_ref = etor.worker_refs[worker_ix]
    while !is_stopping(etor)
        #info("worker_handler!($worker_ix): waiting....")
        Base.acquire(etor.worker_onhold[worker_ix])
        remotecall_fetch(proc, Base.call, worker_ref)
        process_fitness(etor, worker_ix)
        #info("worker_handler!($worker_ix): notify job #$job_id done....")
    end
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
    if force || isnafitness(fitness(candi), fitness_scheme(etor.archive))
        if length(etor.waiting_candidates) >= queue_capacity(etor) && !wait
            #info("async_update_fitness(): queue is full, skip")
            return 0
        end
        #info("async_update_fitness(): waiting to assign job #$(etor.next_job_id)")
        Base.acquire(etor.fitness_slots)
        #info("async_update_fitness(): initial slot_state: $(etor.worker2job), $(etor.fitness_slots.curr_cnt)")
        lock(etor.job_assignment)
        worker_ix = findfirst(etor.worker2job, 0)
        if worker_ix == 0
            error("Cannot find a worker to put a job to")
        end
        etor.worker2job[worker_ix] = job_id = etor.next_job_id
        etor.next_job_id += 1
        copy!(etor.shared_params[worker_ix], candi.params) # share candidate with the workers
        #info("async_update_fitness(): assigning job #$job_id to worker #$worker_ix")
        etor.waiting_candidates[job_id] = candi
        unlock(etor.job_assignment)
        #info("async_update_fitness(): assigned job #$job_id to worker #$worker_ix")
        Base.release(etor.worker_onhold[worker_ix])
        yield() # dispatch the job ASAP
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
            wait(etor)
        end
    end
    if n_pending > 0
        throw(InternalError("Fitnesses not evaluated ($job_ids)"))
    end
    return candidates
end

"""
    wait(etor::ParallelEvaluator)

    Wait until any queued fitness evaluation is complete.
"""
Base.wait(etor::ParallelEvaluator) = wait(etor.fitness_slots.cond_wait)

# FIXME it's not efficient to calculate fitness like that with `ParallelEvaluator`
function fitness{F,FA}(params::Individual, etor::ParallelEvaluator{F,FA})
    candi = Candidate{FA}(params)
    job_id = async_update_fitness(etor, candi, wait=true)
    while !is_stopping(etor) &&
          !(isempty(etor.waiting_candidates) && isempty(etor.unclaimed_candidates))
        #info("fitness(): wait()")
        if isready(etor, job_id)
            return fitness(candi)
        end
        wait(etor)
    end
    throw(InternalError("Fitness not evaluated"))
end
