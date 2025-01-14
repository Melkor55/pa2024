from mpi4py import MPI

def My_MPI_Broadcast(buffer, root, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    mask = 1

    while mask < size:
        if rank < mask:
            if rank + mask < size:
                comm.send(buffer, dest=rank + mask)
        elif rank < 2 * mask:
            buffer = comm.recv(source=rank - mask)
        mask <<= 1
    return buffer

def My_MPI_Reduce(sendbuf, recvbuf, op, root, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    mask = 1

    recvbuf[:] = sendbuf[:]
    while mask < size:
        if rank < mask:
            if rank + mask < size:
                tempbuf = comm.recv(source=rank + mask)
                for i in range(len(recvbuf)):
                    recvbuf[i] += tempbuf[i]  # Assuming MPI_SUM operation
        elif rank < 2 * mask:
            comm.send(recvbuf, dest=rank - mask)
        mask <<= 1
    return recvbuf

def My_MPI_Barrier(comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    mask = 1

    while mask < size:
        if rank < mask:
            if rank + mask < size:
                comm.send(None, dest=rank + mask)
                comm.recv(source=rank + mask)
        elif rank < 2 * mask:
            comm.recv(source=rank - mask)
            comm.send(None, dest=rank - mask)
        mask <<= 1

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
