# -*- coding: utf-8 -*-

"""
Module et_ppmd.timestep 
=================================================================

A module for the time evolution of a set of atoms

"""

class VelocityVerlet:
	def __init__(self,rx,ry,vx,vy,ax,ay):
		"""Velocity Verlet algorithm

		The current time is updated in step_4, when all quantities are known at t
		"""

		self.rx = rx # position
		self.ry = ry
		self.vx = vx # velocity
		self.vy = vy
		self.ax = ax # acceleration
		self.ay = ay
		self.t = 0.0


	def step_12(self,dt):
		"""Step 1 and 2 of the velocity Verlet algorithm.

		Compute the positions at t+dt.

		:param float dt: timestep
		"""
		# Step 1: compute velocities at midstep (t+dt/2) using the current accelerations:
		self.vx_midstep = self.vx + (0.5*dt)*self.ax
		self.vy_midstep = self.vy + (0.5*dt)*self.ay

		# Step 2: compute positions at next step (t+dt) using the midstep velocities:
		self.rx += self.vx_midstep*dt
		self.ry += self.vy_midstep*dt


	def step_4(self, dt):
		"""Step 4 of the velocity Verlet algorithm.

		Compute the velocities at t+dt.

		:param float dt: timestep
		"""
		# Step 4: compute velocities at next step (t+dt)
		self.vx = self.vx_midstep + self.ax * (0.5*dt)
		self.vy = self.vy_midstep + self.ay * (0.5*dt)

		# Update the current time:
		self.t += dt
