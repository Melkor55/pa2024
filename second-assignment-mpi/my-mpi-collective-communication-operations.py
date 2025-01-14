from mpi4py import MPI

##########################################################################################
def My_MPI_Broadcast(buffer, root, comm):
    rank = comm.Get_rank()  # Get the rank of the current process
    size = comm.Get_size()  # Get the total number of processes
    mask = 1  # Initialize the mask to 1

    while mask < size:  # Loop until the mask is less than the number of processes
        if rank < mask:  # If the rank is less than the mask
            if rank + mask < size:  # If the rank plus the mask is less than the number of processes
                comm.send(buffer, dest=rank + mask)  # Send the buffer to the process with rank + mask
        elif rank < 2 * mask:  # If the rank is between mask and 2 * mask
            buffer = comm.recv(source=rank - mask)  # Receive the buffer from the process with rank - mask
        mask <<= 1  # Left shift the mask by 1 (equivalent to multiplying by 2)
    return buffer  # Return the buffer
##########################################################################################

##########################################################################################
def My_MPI_Reduce(sendbuf, recvbuf, op, root, comm):
    rank = comm.Get_rank()  # Get the rank of the current process
    size = comm.Get_size()  # Get the total number of processes
    recvbuf[:] = sendbuf[:]  # Initialize recvbuf with sendbuf

    mask = 1  # Start with mask = 1
    while mask < size:
        partner = rank ^ mask  # XOR to find the communication partner
        if partner < size:  # Ensure partner is within valid range
            if rank < partner:  # Receive from higher ranks
                tempbuf = comm.recv(source=partner)
                for i in range(len(recvbuf)):  # Apply operation (assumes MPI_SUM)
                    recvbuf[i] += tempbuf[i]
            else:  # Send to lower ranks
                comm.send(recvbuf, dest=partner)
        mask <<= 1  # Double the mask
    return recvbuf if rank == root else None
##########################################################################################

##########################################################################################
def My_MPI_Barrier(comm):
    rank = comm.Get_rank()  # Get the rank of the current process
    size = comm.Get_size()  # Get the total number of processes
    mask = 1  # Initialize the mask to 1

    while mask < size:  # Loop until the mask is less than the number of processes
        if rank < mask:  # If the rank is less than the mask
            if rank + mask < size:  # If the rank plus the mask is less than the number of processes
                comm.send(None, dest=rank + mask)  # Send a message to the process with rank + mask
                comm.recv(source=rank + mask)  # Receive a message from the process with rank + mask
        elif rank < 2 * mask:  # If the rank is between mask and 2 * mask
            comm.recv(source=rank - mask)  # Receive a message from the process with rank - mask
            comm.send(None, dest=rank - mask)  # Send a message to the process with rank - mask
        mask <<= 1  # Left shift the mask by 1 (equivalent to multiplying by 2)
##########################################################################################


##########################################################################################
# Test the custom collective communication operations with MPI
##########################################################################################
comm = MPI.COMM_WORLD  # Get the global communicator
rank = comm.Get_rank()  # Get the rank of the current process

# Test My_MPI_Broadcast
data = rank if rank == 0 else None  # Initialize data to the rank if the process is the root, otherwise None
data = My_MPI_Broadcast(data, 0, comm)  # Broadcast the data from the root process
print(f"Process {rank} received data {data}")  # Print the received data

# Test My_MPI_Reduce
data = [rank] * 5  # Initialize data to a list of the rank repeated 5 times
result = [0] * 5  # Initialize the result to a list of zeros
result = My_MPI_Reduce(data, result, MPI.SUM, 0, comm)  # Reduce the data using the sum operation
if rank == 0:  # If the process is the root
    print(f"Reduced result: {result}")  # Print the reduced result

# Test My_MPI_Barrier
My_MPI_Barrier(comm)  # Synchronize all processes
print(f"Process {rank} reached the barrier")  # Print a message indicating the process reached the barrier

MPI.Finalize()  # Finalize the MPI environment
##########################################################################################