!-------------------------------------------------------------------------------------------------
! Fortran source code for module et_ppmd.coref90
!-------------------------------------------------------------------------------------------------
! Remarks:
!   . Enter Python documentation for this module in ``./coref90.rst``.
!     You might want to check the f2py output for the interfaces of the C-wrapper functions.
!     It will be autmatically included in the et_ppmd documentation.
!   . Documument the Fortran routines in this file. This documentation will not be included
!     in the et_ppmd documentation (because there is no recent sphinx
!     extension for modern fortran.

! VERBOSE is for debugging purposes
!#define VERBOSE
! preprocessor directives must start at the beginning of the line in Fortran code

!-------------------------------------------------------------------------------------------------
function force_factor(rij2)
  ! Compute the force factor. Multiply with the interatomic vector to obtain the force vector.
    implicit none
  !-------------------------------------------------------------------------------------------------
  ! subprogram parameters
    real*8 ,intent(in) ::rij2           ! interatomic distance squqred
    real*8             ::force_factor   ! result
  !-------------------------------------------------------------------------------------------------
  ! local variables
    real*8 :: rm2, rm6
  !-------------------------------------------------------------------------------------------------
    rm2 = 1.0/rij2
    rm6 = (rm2*rm2*rm2)
    force_factor = (1.0 - 2.0*rm6)*rm6*rm2*6.0
endfunction

!-------------------------------------------------------------------------------------------------
subroutine computeForces(x,y,vlsz,vlst,fx,fy,n,m)
  ! Compute the forces from the atom positions (x,y) and the Verlet list vl.
  ! See the VerletList class in et_ppmd/verlet.py for an explanation of the
  ! data structure (a 2D integer array.
  !
    implicit none
  !-------------------------------------------------------------------------------------------------
  ! subprogram parameters
    integer*4              , intent(in)    :: n     ! number of atoms
    integer*4              , intent(in)    :: m     ! length of the linearized verlet list = total number of pairs
    real*8   , dimension(n), intent(in)    :: x,y   ! atom positions
    integer*8, dimension(n), intent(in)    :: vlsz  ! size of the individual Verlet lists
    integer*8, dimension(m), intent(in)    :: vlst  ! linearized Verlet lists
    real*8   , dimension(n), intent(inout) :: fx,fy ! atom forces
    ! intent is inout because we do not want to return an array to avoid needless copying
  !-------------------------------------------------------------------------------------------------
  ! local variables
    integer*4 :: i,nb0,nb,j
    integer*4 :: n_neighbours_i
    real*8    :: xij,yij,rij2,ff,ffx,ffy,force_factor
  !------------------------------------------------------------------------------------------------
  ! Zero the forces
    do i=1,n
        fx(i) = 0.0
    end do
    do i=1,n
        fy(i) = 0.0
    end do
#if defined VERBOSE
    do i=1,m
        write (*,*) vlst(i)
    end do
#endif
  ! Compute the interatomic forces and add them to fx and fy
    nb0 = 0 ! offset of verlet list of atom i in the linearized verlet list
  ! Note that Fortran indexes arrays counting from 1 to n, whereas Python, C and C++
  ! count from 0 to n-1.
    do i=1,n
#if defined VERBOSE
        write (*,*) 'coref90:','nb0=',nb0
#endif
        n_neighbours_i = vlsz(i)
        do nb = nb0+1, nb0+n_neighbours_i
#if defined VERBOSE
            write (*,*) 'coref90:','nb=',nb
#endif
            j = vlst(nb) + 1 ! + 1 since Fortran starts counting at 1, and the python atom indices are 0-based
#if defined VERBOSE
            write (*,*) 'coref90:',i,nb,j
#endif
            xij = x(j) - x(i)
            yij = y(j) - y(i)
            rij2 = xij*xij + yij*yij
            ff = force_factor(rij2)
            ffx = ff*xij
            ffy = ff*yij
            fx(i) = fx(i) + ffx
            fy(i) = fy(i) + ffy
            fx(j) = fx(j) - ffx
            fy(j) = fy(j) - ffy
        end do
        nb0 = nb0 + n_neighbours_i
    end do
end subroutine computeForces

!-------------------------------------------------------------------------------------------------
subroutine velocity_verlet_12(dt,rx,ry,vx,vy,ax,ay,vx_midstep,vy_midstep,n)
  ! Compute the forces from the atom positions (x,y) and the Verlet list vl.
  ! See the VerletList class in et_ppmd/verlet.py for an explanation of the
  ! data structure (a 2D integer array.
  !
    implicit none
  !-------------------------------------------------------------------------------------------------
  ! subprogram parameters
    real*8                 , intent(in)    :: dt    ! timestep
    integer*4              , intent(in)    :: n     ! number of atoms
    real*8   , dimension(n), intent(inout) :: rx,ry ! atom positions
    real*8   , dimension(n), intent(in)    :: vx,vy ! atom velocities
    real*8   , dimension(n), intent(in)    :: ax,ay ! atom accelerations
    real*8   , dimension(n), intent(inout) :: vx_midstep,vy_midstep ! midstep atom velocities
  !-------------------------------------------------------------------------------------------------
  ! local variables
    integer*4 :: i
    real*8    :: halfstep
  !------------------------------------------------------------------------------------------------
  ! Step 1: compute velocities at midstep (t+dt/2) using the current accelerations:
  !    self.vx_midstep = self.vx + (0.5*dt)*self.ax
  !    self.vy_midstep = self.vy + (0.5*dt)*self.ay
  ! Step 2: compute positions at next step (t+dt) using the midstep velocities:
  !     self.rx += self.vx_midstep*dt
  !     self.ry += self.vy_midstep*dt
  ! first all work on x components, then on y-components

    halfstep = 0.5*dt;

    do i=1,n
        vx_midstep(i) = vx(i) + halfstep*ax(i)
        rx(i) = rx(i)+ vx_midstep(i) * dt
    end do

    do i=1,n
        vy_midstep(i) = vy(i) + halfstep*ay(i)
        ry(i) = ry(i) + vy_midstep(i) * dt
    end do

end subroutine velocity_verlet_12

!-------------------------------------------------------------------------------------------------
subroutine velocity_verlet_4(dt,rx,ry,vx,vy,ax,ay,vx_midstep,vy_midstep,n)
  ! Compute the forces from the atom positions (x,y) and the Verlet list vl.
  ! See the VerletList class in et_ppmd/verlet.py for an explanation of the
  ! data structure (a 2D integer array.
  !
    implicit none
  !-------------------------------------------------------------------------------------------------
  ! subprogram parameters
    real*8                 , intent(in)    :: dt    ! timestep
    integer*4              , intent(in)    :: n     ! number of atoms
    real*8   , dimension(n), intent(in)    :: rx,ry ! atom positions
    real*8   , dimension(n), intent(inout) :: vx,vy ! atom velocities
    real*8   , dimension(n), intent(in)    :: ax,ay ! atom accelerations
    real*8   , dimension(n), intent(in)    :: vx_midstep,vy_midstep ! midstep atom velocities
  !-------------------------------------------------------------------------------------------------
  ! local variables
    integer*4 :: i
    real*8    :: halfstep
  !------------------------------------------------------------------------------------------------
  ! Step 4: compute velocities at next step (t+dt)
  ! 	self.vx = self.vx_midstep + self.ax * (0.5*dt)
  ! 	self.vy = self.vy_midstep + self.ay * (0.5*dt)

    halfstep = 0.5*dt
    
    do i=1,n
        vx(i) = vx_midstep(i) + halfstep*ax(i)
    end do

    do i=1,n
        vy(i) = vy_midstep(i) + halfstep*ay(i);
    end do

end subroutine velocity_verlet_4
