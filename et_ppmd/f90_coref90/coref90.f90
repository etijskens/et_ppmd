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
subroutine computeForces(x,y,vl,fx,fy,n,m)
  ! Compute the forces from the atom positions (x,y) and the Verlet list vl.
  ! See the VerletList class in et_ppmd/verlet.py for an explanation of the
  ! data structure (a 2D integer array.
  !
    implicit none
  !-------------------------------------------------------------------------------------------------
  ! subprogram parameters
    integer*4                , intent(in)    :: n ! number of atoms
    integer*4                , intent(in)    :: m ! maximum number of neighbours
    real*8   , dimension(n)  , intent(in)    :: x,y
    integer*4, dimension(m,n), intent(in)    :: vl ! Verlet lists
    ! Note that Fortran has column major storage order, and in Python the array was created with
    ! row major storage order. Therefor, we must flip the order of the indices.
    real*8   , dimension(n)  , intent(inout) :: fx,fy
    ! intent is inout because we do not want to return an array to avoid needless copying
  !-------------------------------------------------------------------------------------------------
  ! local variables
    integer*4 :: i,nb,j
    integer*4 :: n_neighbours
    real*8    :: xij,yij,rij2,ff,ffx,ffy,force_factor
  !------------------------------------------------------------------------------------------------
  ! Zero the forces
    do i=1,n
        fx(i) = 0.0
    end do
    do i=1,n
        fy(i) = 0.0
    end do
  ! Compute the interatomic forces and add them to fx and fy
    do i=1,n
        n_neighbours = vl(1,i) ! 1 since Fortran starts counting at 1
        do nb=1,n_neighbours
            j = vl(1+nb,i) + 1 ! + 1 since Fortran starts counting at 1, and the python atom indices are 0-based
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
    end do
end subroutine computeForces
