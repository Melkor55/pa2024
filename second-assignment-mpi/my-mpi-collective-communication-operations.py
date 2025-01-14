from mpi4py import MPI

def My_MPI_Broadcast(buffer, root, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == root:
        for i in range(size):
            if i != root:
                comm.send(buffer, dest=i)
    else:
        buffer = comm.recv(source=root)
    return buffer

def My_MPI_Reduce(sendbuf, recvbuf, op, root, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == root:
        recvbuf[:] = sendbuf[:]
        for i in range(size):
            if i != root:
                tempbuf = comm.recv(source=i)
                for j in range(len(recvbuf)):
                    recvbuf[j] += tempbuf[j]
    else:
        comm.send(sendbuf, dest=root)
    return recvbuf

def My_MPI_Barrier(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()

    for i in range(size):
        if i != rank:
            comm.send(None, dest=i)
            comm.recv(source=i)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Test My_MPI_Broadcast
data = rank if rank == 0 else None
data = My_MPI_Broadcast(data, 0, comm)
print(f"Process {rank} received data {data}")

# Test My_MPI_Reduce
data = [rank] * 5
result = [0] * 5
result = My_MPI_Reduce(data, result, MPI.SUM, 0, comm)
if rank == 0:
    print(f"Reduced result: {result}")

# Test My_MPI_Barrier
My_MPI_Barrier(comm)
print(f"Process {rank} reached the barrier")

MPI.Finalize()
