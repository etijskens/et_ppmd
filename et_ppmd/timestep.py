# -*- coding: utf-8 -*-

"""
Module et_ppmd.timestep 
=================================================================

A module for the time evolution of a set of atoms

"""
import numpy as np
import et_ppmd.corecpp
import et_ppmd.coref90

class VelocityVerlet:
	def __init__(self,rx,ry,vx,vy,ax,ay,impl='python'):
		"""Velocity Verlet algorithm

		:param impl: 'python'*|'cpp'|'f90': implementation used
		The current time is updated in step_4, when all quantities are known at t
		"""

		self.rx = rx # position
		self.ry = ry
		self.vx = vx # velocity
		self.vy = vy
		self.ax = ax # acceleration
		self.ay = ay
		self.n_atoms = rx.shape[0]
		self.t = 0.0
		if not impl in ['python','cpp','f90']:
			raise ValueError(f"Invalid implementation {impl}.")
		self.impl = impl
		if self.impl != "python":
			self.vx_midstep = np.empty((self.n_atoms,), dtype=float)
			self.vy_midstep = np.empty((self.n_atoms,), dtype=float)


	def step_12(self,dt):
		"""Step 1 and 2 of the velocity Verlet algorithm.

		Compute the positions at t+dt.

		:param float dt: timestep
		"""
		if self.impl=='python':
			# Step 1: compute velocities at midstep (t+dt/2) using the current accelerations:
			self.vx_midstep = self.vx + (0.5*dt)*self.ax
			self.vy_midstep = self.vy + (0.5*dt)*self.ay

			# Step 2: compute positions at next step (t+dt) using the midstep velocities:
			self.rx += self.vx_midstep*dt
			self.ry += self.vy_midstep*dt

		elif self.impl=='cpp':
			# print(self.rx)
			# print(self.vx)
			# print(self.ax)
			# print(self.vx_midstep)
			# print(self.ry)
			# print(self.vy)
			# print(self.ay)
			# print(self.vy_midstep)

			et_ppmd.corecpp.velocity_verlet_12( dt
											 , self.rx, self.ry
											 , self.vx, self.vy
											 , self.ax, self.ay
										     , self.vx_midstep, self.vy_midstep
											 )
		elif self.impl=='f90':
			et_ppmd.coref90.velocity_verlet_12( dt
											 , self.rx, self.ry
											 , self.vx, self.vy
											 , self.ax, self.ay
										     , self.vx_midstep, self.vy_midstep
											 )

	def step_4(self, dt):
		"""Step 4 of the velocity Verlet algorithm.

		Compute the velocities at t+dt.

		:param float dt: timestep
		"""
		if self.impl=='python':
			# Step 4: compute velocities at next step (t+dt)
			self.vx = self.vx_midstep + self.ax * (0.5*dt)
			self.vy = self.vy_midstep + self.ay * (0.5*dt)
		elif self.impl=='cpp':
			et_ppmd.corecpp.velocity_verlet_4( dt
											, self.rx, self.ry
											, self.vx, self.vy
											, self.ax, self.ay
											, self.vx_midstep, self.vy_midstep
											)

		elif self.impl=='f90':
			et_ppmd.coref90.velocity_verlet_4( dt
											, self.rx, self.ry
											, self.vx, self.vy
											, self.ax, self.ay
											, self.vx_midstep, self.vy_midstep
											)
		# Update the current time:
		self.t += dt
