

def main():
    from resonet.sims.main import args, run
    from libtbx.mpi4py import MPI
    COMM = MPI.COMM_WORLD
    import os
    import time
    args_parsed = args()

    rank_offset = args_parsed.rankOffset
    if rank_offset > 0 and args_parsed.totalRanks is None:
        if COMM.rank == 0:
            raise ValueError(
                "--rankOffset requires --totalRanks to be set explicitly. "
                f"E.g. --rankOffset {rank_offset} --totalRanks {rank_offset + COMM.size}"
            )
    total_ranks = args_parsed.totalRanks if args_parsed.totalRanks is not None else COMM.size

    # generate random seeds covering all effective ranks
    seed_time = args_parsed.seed
    if args_parsed.seed is None:
        if COMM.rank == 0:
            seed_time = int(time.time())
        seed_time = COMM.bcast(seed_time)
    seeds = [seed_time + r for r in range(total_ranks)]

    # create output directory
    if COMM.rank == 0:
        if not os.path.exists(args_parsed.outdir):
            os.makedirs(args_parsed.outdir)
    COMM.barrier()

    from simtbx.diffBragg.device import DeviceWrapper
    effective_rank = COMM.rank + rank_offset
    dev_id = effective_rank % args_parsed.ngpu
    import numpy as np
    gvec = None
    if COMM.rank==0:
        if args_parsed.randAxis:
            gvec = np.random.normal(0,1,3)
    gvec = COMM.bcast(gvec)
    # TODO: remove this, its for debugging!
    #gvec = np.array([-0.11714061589265543, 0.48394403574869455, 0.8672232967186454])
    with DeviceWrapper(dev_id) as _:
        try:
            run(args_parsed, seeds, effective_rank, total_ranks, gvec=gvec)
        except Exception:
            import sys
            import traceback
            tb_s = traceback.format_exc()
            print(f"RANK {effective_rank}: FATAL ERROR\n{tb_s}", file=sys.stderr, flush=True)
            err_file = os.path.join(args_parsed.outdir, "rank%d_failure.err" % effective_rank)
            try:
                with open(err_file, "w") as o:
                    o.write(tb_s)
            except OSError as file_err:
                print(f"RANK {effective_rank}: could not write {err_file}: {file_err}", file=sys.stderr, flush=True)
            COMM.Abort(1)


if __name__=="__main__":
    main()
