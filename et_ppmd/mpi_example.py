# -*- coding: utf-8 -*-

"""
script et_ppmd.mpi_example
=================================================================

mpi4py example script

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!! Make sure to add mpi4py as a dependency and pytest-mpi as a !!!
!!! development dependency to your project						!!!
!!!  (.venv)> poetry add mpi4py									!!!
!!!  (.venv)> poetry add mpi4py --dev 							!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

To run this script in parallel on e.g. 2 processes, enter this command
in a terminal::

	(.venv)> mpirun -np 2 python et_ppmd/mpi_example.py

This is what the output looks like::

	(.venv)> mpirun -np 2 python et_ppmd/mpi_example.py
	rank 0/2:
	rank 0/2: about to start sending data to rank 1
	rank 1/2:
	rank 1/2: about to start receiving data from rank 0
	rank 1/2: I might be doing something else, while the data are being received ...
	rank 0/2: I might be doing something else, while the data are being send ...
	rank 0/2: I am sure that rank 1 has received the data now
	rank 0/2: -*# done #*-
	rank 1/2: the data are received. I can start processing them.
	rank 1/2: data={'a': 7, 'b': 3.14}
	rank 1/2: -*# done #*-

The order of the lines might differ from one run to another. The print
statements of rank will be in the correct order, for rank 0::

	rank 0/2:
	rank 0/2: about to start sending data to rank 1
	rank 0/2: I might be doing something else, while the data are being send ...
	rank 0/2: I am sure that rank 1 has received the data now
	rank 0/2: -*# done #*-

and for rank 1::

	rank 1/2:
	rank 1/2: about to start receiving data from rank 0
	rank 1/2: I might be doing something else, while the data are being received ...
	rank 1/2: the data are received. I can start processing them.
	rank 1/2: data={'a': 7, 'b': 3.14}
	rank 1/2: -*# done #*-

Both sets of statements may interleave differently.

"""


from mpi4py import MPI

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank() # the mpi rank of this process
mpi_size = comm.Get_size() # total number of collaborating processes

def mpi_message(msg=''):
	"""Preceed a mpi_message with rank information.

	:param str msg: mpi_message.
	:return: str
	"""
	return f'rank {mpi_rank}/{mpi_size}: {msg}'


if __name__=='__main__':
	# print a mpi_message from each process
	print(mpi_message())
	if mpi_rank == 0:
		# non-blocking send to rank 1
		data = {'a': 7, 'b': 3.14}
		print(mpi_message('about to start sending data to rank 1'))
		req = comm.isend(data, dest=1, tag=11)
		print(mpi_message('I might be doing something else, while the data are being send ...'))
		req.wait()
		print(mpi_message('I am sure that rank 1 has received the data now'))

	elif mpi_rank == 1:
		print(mpi_message('about to start receiving data from rank 0'))
		req = comm.irecv(source=0, tag=11)
		print(mpi_message('I might be doing something else, while the data are being received ...'))
		data = req.wait()
		print(mpi_message('the data are received. I can start processing them.'))
		print(mpi_message(f'data={data}'))

	print(mpi_message('-*# done #*-'))