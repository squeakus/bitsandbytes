	program truss

c   This program performs finite element analysis
c   by reading in the data, doing assembling, and
c   then solving the linear system.
c
c    SLFFEA source file
c    Version:  1.0
c    Copyright (C) 1999  San Le
c
c    The source code contained in this file is released under the
c    terms of the GNU Library General Public License.
c
c
c
c	Updated 5/25/00 
c
c       max 100 elements
c       max 100 nodes
c       max 300 degress of freedom
c

	include "tsconst.hf"

        integer i, j, file1, file2, temp, bandmax
        integer id(mxnode,ndof), lm(mxel,neqel), idiag(mxdof)
        integer connect(mxnode,npel), elmatl(mxel), dum
        integer bcnumfixx, bcnumfixy, bcnumfixz
        integer bcfixx(mxnode), bcfixy(mxnode), bcfixz(mxnode)
        double precision matlE(mxel), matlmass(mxel), matlarea(mxel)
        double precision fpointx, fpointy, fpointz
        double precision ncoord(mxnode,nsd), force(mxdof)
	double precision U(mxdof), A(mxkmat)
        character *20, name
        integer numnp, numel, nmat, nmode, dof, neqn, flag

	common/var1/numnp, numel, nmat, nmode, dof, neqn, flag

    	write(6,*) "What is the name of the file containing the"
    	write(6,*) "truss structural data? (example: tsp413)"
    	read(5,*) name

c    10 contains all the structural data

	file1 = 10
	file2 = 20

      	open(unit=file1,file=name,status='unknown')

        read(file1,*) 
        read(file1,*) numel,numnp,nmat,nmode
        dof=numnp*ndof

        call bzero1(U, dof)
        call bzero1(force, dof)

	call reader(connect, ncoord, elmatl, matlE, matlmass,
     &  matlarea, force, U, bcnumfixx, bcnumfixy, bcnumfixz,
     &	bcfixx, bcfixy, bcfixz, file1)

        call formid( id, bcnumfixx, bcnumfixy, bcnumfixz, bcfixx,
     &	bcfixy, bcfixz)

        write(6,*)
        write(6,*) " This is the id matrix "
        write(6,*)
        do 100 i = 1, numnp
                write(6,9) i,(id(i,j), j = 1, ndof)
100     continue
9       format(" node(",i4,")",3(i5,1x))

        call formlm( id, connect, lm )

        write(6,*)
        write(6,*) " This is the lm matrix "
        write(6,*)
        do 200 i = 1, numel
                write(6,10) i,(lm(i,j), j = 1, neqel)
200     continue
10      format(" element(",i4,")",6(i5,1x))

        call diag( lm, idiag )

        write(6,*)
        write(6,*)
        write(6,*) " This is the idiag matrix "
        write(6,*)
        write(6,11) 1, idiag(1)
	temp = 0
	bandmax = 0
        do 300 i = 2, neqn
            write(6,11) i,idiag(i)
	    temp = idiag(i)-idiag(i-1)
            if( temp .gt. bandmax ) bandmax = temp 
300     continue
        write(6,*)
        write(6,*) "  This is the maximum bandwidth " 
        write(6,*)
        write(6,11) bandmax
11      format(" node(",i4,")",i5)

        flag = 1
        call assemble(connect, ncoord, elmatl, matlE, matlarea, force,
     &U, id, lm, idiag, A, bcnumfixx, bcnumfixy, bcnumfixz, bcfixx,
     &bcfixy, bcfixz)

        write(6,*)
        write(6,*)
        write(6,*) " This is the force matrix "
        write(6,*)
        do 400 i = 1, neqn
            write(6,12) i,force(i)
400     continue
12      format(" dof ",i5,"   ",f18.6,1x)

c  Perform LU Crout decompostion on the system

        call decomp(A,idiag,neqn)

c  Solve the system

        call solve(A,force,idiag,neqn)

        write(6,*)
        write(6,*) " This is the solution to the problem "
        write(6,*)
        do 500 i = 1, numnp
           do 501 j = 1, ndof
                if( id(i,j) .gt. -1 ) then
                        U(ndof*(i-1)+j) = force(id(i,j))
                end if
501        continue
500     continue
        do 600 i = 1, numnp
                if( id(i,1) .gt. -1 ) then
                        write(6,13) i, U(ndof*(i-1)+1)
                end if
                if( id(i,2) .gt. -1 ) then
                        write(6,14) i, U(ndof*(i-1)+2)
                end if
                if( id(i,3) .gt. -1 ) then
                        write(6,15) i, U(ndof*(i-1)+3)
                end if
600     continue
13      format(' node ',i3,' x   ',f14.6,1x)
14      format(' node ',i3,' y   ',f14.6,1x)
15      format(' node ',i3,' z   ',f14.6,1x)

c  Calculate the reaction forces
        flag = 2
        call bzero1(force,dof)
        write(6,*)
        write(6,*)
        write(6,*) " These are the axial displacements and forces "
        write(6,*)

        call assemble(connect, ncoord, elmatl, matlE, matlarea, force,
     &U, id, lm, idiag, A, bcnumfixx, bcnumfixy, bcnumfixz, bcfixx,
     &bcfixy, bcfixz)

        write(6,*)
        write(6,*) " These are the reaction forces "
        write(6,*)
        do 700 i = 1, numnp
                if( id(i,1) .lt. 0 ) then
                        write(6,16) i, force(ndof*(i-1)+1)
                end if
                if( id(i,2) .lt. 0 ) then
                        write(6,17) i, force(ndof*(i-1)+2)
                end if
                if( id(i,3) .lt. 0 ) then
                        write(6,18) i, force(ndof*(i-1)+3)
                end if
700     continue
16      format(' node ',i3,' x   ',f18.6,1x)
17      format(' node ',i3,' y   ',f18.6,1x)
18      format(' node ',i3,' z   ',f18.6,1x)

        write(6,*)
        write(6,*) "               These are the updated coordinates "
        write(6,*)
        write(6,*) "                  x               y             z "
        write(6,*)

        write(11,*) numnp, 1
        do 800 i = 1, numnp
                fpointx = ncoord(i,1) + U(ndof*(i-1)+1)
                fpointy = ncoord(i,2) + U(ndof*(i-1)+2)
                fpointz = ncoord(i,3) + U(ndof*(i-1)+3)
                write(6,19) i,fpointx,fpointy,fpointz
800     continue
19      format(' node ',i3,3(1x,f16.9))
        write(6,*)

	end

c  This library function reads in data for a finite element 
c  program which does analysis on a truss 
c       Written by San Le 
c
c	Updated on 5/25/00

	subroutine reader(connect, ncoord, elmatl, matlE,
     &	matlmass, matlarea, force, U, bcnumfixx, bcnumfixy,
     &	bcnumfixz, bcfixx, bcfixy, bcfixz, file1)

	include "tsconst.hf"

        integer i,j,dum
	character text

        integer file1
        integer bcnumfixx, bcnumfixy, bcnumfixz
        integer bcfixx(mxnode), bcfixy(mxnode), bcfixz(mxnode)
        integer connect(mxnode,npel), elmatl(mxel)
        double precision ncoord(mxnode,nsd), force(mxdof), U(mxdof)
        double precision matlE(mxel), matlmass(mxel), matlarea(mxel)
        integer numnp, numel, nmat, nmode, dof, neqn, flag

	common/var1/numnp, numel, nmat, nmode, dof, neqn, flag

        write(6,2000) numel,numnp,nmat,nmode,dof
2000	format('number of elements:',i4,' nodes:',i4,
     &	' materials:',i4,' nmodes:',i4,' dof:',i4)

        read(file1,*)  
        write(6,*) 

        do 100 i = 1, nmat  
           read(file1,*)  dum, matlE(dum+1), matlmass(dum+1),
     &	matlarea(dum+1)
           write(6,20) dum+1, matlE(dum+1), matlmass(dum+1),
     &	matlarea(dum+1)
100     continue
20	format('material (',i3,') Emod, mass, Area ',1x,e12.4,
     &	1x,f9.4, 1x,f9.4)
        read(file1,*) 
        write(6,*) 

        do 200 i = 1, numel  
                read(file1,*)  dum, (connect(dum+1,j), j = 1, 
     &	npel), elmatl(dum+1)
		elmatl(dum+1) = elmatl(dum+1) + 1
		connect(dum+1,1) = connect(dum+1,1) + 1
		connect(dum+1,2) = connect(dum+1,2) + 1
                write(6,21) dum+1, (connect(dum+1,j), j = 1, npel),
     &	elmatl(dum+1)
200     continue
21	format('connectivity for element (',i3,')',2(1x,i3),
     &	' with matl ',i3)
        read(file1,*) 
        write(6,*) 

        do 300 i = 1, numnp  
           read(file1,*) dum, (ncoord(dum+1,j), j = 1, nsd)
           write(6,22) dum+1, (ncoord(dum+1,j), j = 1, nsd)
300     continue
22	format('coordinate (',i5,') coordinates ',3(1x,f9.4))
        read(file1,*) 
        write(6,*) 

        dum= 1
	do 400 i = 1, numnp + 10
        	read(file1,*) bcfixx(dum),
     &	(U(ndof*bcfixx(dum)+1), 
     &	j = 1, (bcfixx(dum)+1)/abs(bcfixx(dum)+1))
        	write(6,23) bcfixx(dum)+1,
     &	(U(ndof*bcfixx(dum)+1), 
     &	j = 1, (bcfixx(dum)+1)/abs(bcfixx(dum)+1))
                if( bcfixx(dum) .lt. -1 ) goto 231
		bcfixx(dum) = bcfixx(dum) + 1
                dum = dum + 1
400     continue
23	format('node (',i4,
     &	') has an x prescribed displacement of: ', f9.5)
231     bcnumfixx=dum
        read(file1,*) 
        write(6,*) 

        dum= 1
	do 500 i = 1, numnp + 10
        	read(file1,*) bcfixy(dum),
     &	(U(ndof*bcfixy(dum)+2), 
     &	j = 1, (bcfixy(dum)+1)/abs(bcfixy(dum)+1))
        	write(6,24) bcfixy(dum)+1,
     &	(U(ndof*bcfixy(dum)+2), 
     &	j = 1, (bcfixy(dum)+1)/abs(bcfixy(dum)+1))
                if( bcfixy(dum) .lt. -1 ) goto 241
		bcfixy(dum) = bcfixy(dum) + 1
                dum = dum + 1
500     continue
24	format('node (',i4,
     &	') has an y prescribed displacement of: ', f9.5)
241     bcnumfixy=dum
        read(file1,*) 
        write(6,*)

        dum= 1
	do 600 i = 1, numnp + 10
        	read(file1,*) bcfixz(dum),
     &	(U(ndof*bcfixz(dum)+3), 
     &	j = 1, (bcfixz(dum)+1)/abs(bcfixz(dum)+1))
        	write(6,25) bcfixz(dum)+1,
     &	(U(ndof*bcfixz(dum)+3), 
     &	j = 1, (bcfixz(dum)+1)/abs(bcfixz(dum)+1))
                if( bcfixz(dum) .lt. -1 ) goto 251
		bcfixz(dum) = bcfixz(dum) + 1
                dum = dum + 1
600     continue
25	format('node (',i4,
     &	') has an y prescribed displacement of: ', f9.5)
251     bcnumfixz=dum
        read(file1,*) 
        write(6,*) 

	dum = 0
	do 700 i = 1, numnp + 10
           read(file1,*) dum, (force(ndof*dum+j), 
     &j = 1, ndof*(dum+1)/abs(dum+1))
           write(6,26) dum+1, (force(ndof*dum+j), 
     &j = 1, ndof*(dum+1)/abs(dum+1))
           if( dum .lt. -1 ) goto 261 
700     continue
26	format('force vector for node: (',i4,')',3(1x,e13.4))
261     write(6,*)

	return
	end

c  This utility function assembles the id and lm arrays for a finite
c  element program which does analysis on a 2 node truss element
c
c               Updated 5/25/00

        subroutine formid( id, bcnumfixx, bcnumfixy, bcnumfixz, bcfixx,
     &	bcfixy, bcfixz)


c  Assembly of the id array(the matrix which determines
c  the degree of feedom by setting fixed nodes = -1)

	include "tsconst.hf"

        integer i, j, counter
        integer id(mxnode,ndof)
        integer bcnumfixx, bcnumfixy, bcnumfixz
        integer bcfixx(mxnode), bcfixy(mxnode), bcfixz(mxnode)

        integer numnp, numel, nmat, nmode, dof, neqn, flag

	common/var1/numnp, numel, nmat, nmode, dof, neqn, flag

        counter=1

        do 100 i = 1, bcnumfixx-1
           id(bcfixx(i), 1) = -1
100     continue
        do 200 i = 1, bcnumfixy-1
           id(bcfixy(i), 2) = -1
200     continue
        do 300 i = 1, bcnumfixz-1
           id(bcfixz(i), 3) = -1
300     continue
        do 400 i = 1, numnp
           do 401 j = 1, ndof
           if( id(i,j).ne. -1 ) then
                id(i,j) = counter
                counter = counter + 1
           end if
401        continue
400     continue
        neqn=counter - 1
        return
        end

        subroutine formlm( id, connect, lm )

c  Assembly of the lm array(the matrix which gives
c  the degree of feedom per element and node )

	include "tsconst.hf"

        integer id(mxnode,ndof), lm(mxel,neqel)
        integer connect(mxnode,npel)
        integer i,j,k,node
        integer j2

        integer numnp, numel, nmat, nmode, dof, neqn, flag

	common/var1/numnp, numel, nmat, nmode, dof, neqn, flag

        do 100 k = 1, numel
           do 101 j = 1, npel
                node = connect(k, j)
                do 102 i = 1, ndof
                        lm(k , ndof*(j-1) + i) = id(node, i)
102             continue
101        continue
100     continue
        return
        end


c  This utility function assembles the idiag array for a finite
c  element program which does analysis on a 2 node beam
c  element
c
c               Updated 5/25/98

        subroutine diag( lm, idiag )

	include "tsconst.hf"

        integer lm(mxel,neqel), idiag(mxdof)
        integer i,j,k,m,node,minimum,num

        integer numnp, numel, nmat, nmode, dof, neqn, flag

	common/var1/numnp, numel, nmat, nmode, dof, neqn, flag

c  Assembly of the idiag array for the skyline

        do 100 i = 1, neqn
                idiag(i)=i
100     continue

        do 200 k = 1, numel
           minimum=neqn
           do 201 j = 1, npel
                do 202 i = 1, ndof
                        num= lm(k, ndof*(j-1) + i)
                        if(num .gt. -1 ) then
                                minimum = MIN(minimum,num)
                        end if
202             continue
201        continue

           do 203 j = 1, npel
                do 204 i = 1, ndof
                        num= lm(k, ndof*(j-1) + i)
                        if(num .gt. -1 ) then
                                idiag(num) = MIN(idiag(num),minimum)
                        end if
204             continue
203        continue
200     continue
        do 300 i = 2, neqn
                idiag(i) = idiag(i-1)+i - idiag(i)+1
300     continue
        return
        end


c  This library function assembles the stiffness matrix and
c  calculates the reaction forces for a finite element program
c  which does analysis on a truss 
c
c       Written by San Le 
c
c       Updated 1/9/00 

	subroutine assemble(connect, ncoord, elmatl, matlE,
     &	 matlarea, force, U, id, lm, idiag, A, bcnumfixx,
     &	 bcnumfixy, bcnumfixz, bcfixx, bcfixy, bcfixz)

	include "tsconst.hf"

        integer i,i2,j,j2,j3,k,i1,j1,ij,i10,i11,i12,i13,i14,i15
	integer dum0,dum1,check,counter
        integer ijmax,ijabs,locina,lmi,lmj,matlnum
        double precision L, Lx, Ly, Lz, Lsq
        double precision B(sdim,npel), DB(sdim,npel), jacob
	double precision Ktemp(npel,neqel),Kint(npel,npel),Kel(neqel,neqel)
	double precision Klocal(npel,npel), rotate(npel,neqel)
	double precision forceel(neqel), forceax(npel)
	double precision Uel(neqel), Uax(npel)
	double precision x(numint), w(numint)
        double precision EmodXarea

        integer id(mxnode,ndof), lm(mxel,neqel), idiag(mxdof)
	integer bcnumfixx, bcnumfixy, bcnumfixz
	integer bcfixx(mxnode), bcfixy(mxnode), bcfixz(mxnode)
	integer connect(mxnode,npel), elmatl(mxel)
	double precision ncoord(mxnode,nsd), force(mxdof)
	double precision U(mxdof), A(mxkmat)
        double precision matlE(mxel), matlarea(mxel)

        integer numnp, numel, nmat, nmode, dof, neqn, flag

	common/var1/numnp, numel, nmat, nmode, dof, neqn, flag

        call bzero1(A, idiag(neqn)+1)
        call bzero2(rotate,npel,neqel)

        x(1)=0.0
        w(1)=2.0

        do 100 k = 1, numel  

		matlnum = elmatl(k)
        	EmodXarea = matlE(matlnum)*matlarea(matlnum)

                dum0 = connect(k,1)
                dum1 = connect(k,2)

                Lx = ncoord(dum1,1) - ncoord(dum0,1)
                Ly = ncoord(dum1,2) - ncoord(dum0,2)
                Lz = ncoord(dum1,3) - ncoord(dum0,3)

                Lsq = Lx*Lx+Ly*Ly+Lz*Lz
                L = sqrt(Lsq)

		Lx = Lx/L 
		Ly = Ly/L 
		Lz = Lz/L

                jacob = L/2.0

c  Assembly of the rotation matrix 

                rotate(1,1) = Lx
                rotate(1,2) = Ly
                rotate(1,3) = Lz
                rotate(2,4) = Lx
                rotate(2,5) = Ly
                rotate(2,6) = Lz

c  defining the components of an element vector 

                i10=ndof*(dum0-1) + 1
                i11=ndof*(dum0-1) + 2
                i12=ndof*(dum0-1) + 3
                i13=ndof*(dum1-1) + 1
                i14=ndof*(dum1-1) + 2
                i15=ndof*(dum1-1) + 3

                call bzero1(Uel, neqel)
                call bzero2(Kel, neqel, neqel)
                call bzero1(forceel, neqel)
                call bzero1(forceax, npel)
                call bzero2(Ktemp, npel, neqel)
                call bzero2(Kint, npel, npel)

c  The loop below calculates the 1 pointeger of the gaussian
c  integration 
                do 101 j = 1, numint

                    call bzero2(B, sdim, npel)
                    call bzero2(DB, sdim, npel)
                    call bzero2(Klocal, npel, npel)

c  Assembly of the local stiffness matrix 

                    B(1,1) = - 1.0/L
                    B(1,2) = 1.0/L

                    DB(1,1) = EmodXarea*B(1,1)
                    DB(1,2) = EmodXarea*B(1,2)

                    call matXT(B, DB, Klocal, npel, npel, sdim)

                    do 102 j2 = 1 , npel
                        do 1020 j3 = 1 , npel 
                            Kint(j2,j3) = Kint(j2,j3) + 
     &	Klocal(j2,j3)*jacob*w(j)
1020                    continue
102                 continue
101       	continue

c  Put K back to global coordinates 

                call matX(Kint, rotate, Ktemp, npel, neqel, npel)

                call matXT(rotate, Ktemp, Kel, neqel, neqel, npel)

		Uel(1) = U(i10)
		Uel(2) = U(i11)
		Uel(3) = U(i12)
		Uel(4) = U(i13)
		Uel(5) = U(i14)
		Uel(6) = U(i15)

                call matX(Kel, Uel, forceel, neqel, 1, neqel)

		if(flag .eq. 1) then

c  Compute the equivalant nodal forces based on prescribed
c  displacements 

		  force(i10) = force(i10) - forceel(1)
		  force(i11) = force(i11) - forceel(2)
		  force(i12) = force(i12) - forceel(3)
		  force(i13) = force(i13) - forceel(4)
		  force(i14) = force(i14) - forceel(5)
		  force(i15) = force(i15) - forceel(6)

c  Assembly of the global skylined  stiffness matrix 

                  do 103 i = 1, neqel  
                    if( lm(k, i) .gt. -1 ) then
                        do 104 j = i, neqel  
                           if( lm(k, j) .gt. -1 ) then
                                lmi=lm(k, i)
                                lmj=lm(k, j)
                                ijmax = MAX(lmi,lmj)
                                ijabs = abs(lmi-lmj)
                                locina = idiag(ijmax) - ijabs
                                A(locina) = A(locina) + Kel(i,j)
c                                write(6,8) locina, A(locina), Kel(i,j)
                           end if
104                     continue
                    end if
103               continue
		else
c Calculate the element reaction forces 

			force(i10) = force(i10) + forceel(1)
			force(i11) = force(i11) + forceel(2)
			force(i12) = force(i12) + forceel(3)
			force(i13) = force(i13) + forceel(4)
			force(i14) = force(i14) + forceel(5)
			force(i15) = force(i15) + forceel(6)

c  Calculate the element axial forces 

                        write(6,9) k,dum0,dum1

                	call matX(rotate, Uel, Uax, npel, 1,
     &	neqel)
                        write(6,10) Uax(1), Uax(2)

                	call matX(Klocal, Uax, forceax, npel,
     &	1, npel)
			forceax(1) = L*forceax(1)
			forceax(2) = L*forceax(2)

                        write(6,11) forceax(1), forceax(2)
		end if
100     continue
8       format(i3,1x,f14.5,1x,f14.5,1x)
9       format(' element (',i3,')  node ',i3,'       node ',i3)
10      format(' displacement  ',f9.5,'      ',f9.5)
11      format(' force    ',f14.5,' ',f14.5)

	if(flag .eq. 1) then

c  Assembly of the global force matrix 

          counter = 1
          do 200 i = 1, numnp   
            if( id(i,1) .gt. -1 ) then
                force(counter) = force(ndof*(i-1)+1)
                counter = counter + 1
            end if
            if( id(i,2) .gt. -1 ) then
                force(counter) = force(ndof*(i-1)+2)
                counter = counter + 1
            end if
            if( id(i,3) .gt. -1 ) then
                force(counter) = force(ndof*(i-1)+3)
                counter = counter + 1
            end if
200       continue
        end if

	return
	end

c  This subroutine multiplies the matrices A*B
c
c                A(n,p) B(p,m)
c
c       written by San Le
c
c       Updated 11/4/99
c
c
        subroutine matX(A, B, C, n, m, p)

        integer i, j, k, n, m, p
        double precision C(n,m),A(n,p),B(p,m)

        call bzero2(C,n,m)

        do 100 i = 1, n
           do 101 j = 1, m
                do 102 k = 1, p
                   C(i,j) = C(i,j) + A(i,k)*B(k,j)
102             continue
c                write(6,*) "%i %i %10.7f ",i,j,C(i,j)
101        continue
c           write(6,*)
100     continue
        return
        end

c  This subroutine multiplies the matrices A(Transpose)*B
c
c               A(p,n) B(p,m)
c
c               written by San Le
c
c               Updated 11/4/99


        subroutine matXT(A, B, C, n, m, p)

        integer i, j, k, n, m, p
        double precision C(n,m),A(p,n),B(p,m)

        call bzero2(C,n,m)
        do 100 i = 1, n
           do 101 j = 1, m
                do 102 k = 1, p
                    C(i,j) = C(i,j) + A(k,i)*B(k,j)
c                   write(6,*) "%i %i %i ",n,k+1-i-1,n*(k+1)-i-1)
c                   write(6,*) "%i %i %i %i %i %i  ",i,j,k,m*i+j,
c                       n*(k+1)-i-1,m*k+j)

102             continue
c                write(6,*) "%i %i %10.7f ",i,j,*(C+m*i+j)
101        continue
c           write(6,*)
100     continue
        return
        end

c  This subroutine performs skyline decomposition and also calculates
c  the solution of the skylined system.  It is based on Dr. David Benson's 
c  algorithm for ames 232 fall 1996.  
c
c	Implemented by San Le.
c                
c                Updated 6/17/99

c  This program performs LU decomposition on a skylined matrix 

	subroutine decomp( A, idiag, neq)

	include "tsconst.hf"

        integer i,j,k,rk,ri
	integer iloci, ilock, iloc, len
	double precision p, t, dotX

        integer idiag(mxdof), neq
	double precision A(mxkmat)

	external dotX

        do 100 k = 2, neq  
c  Calculate the fist row in column k 

		rk = k + 1 + idiag( k - 1 ) - idiag( k)
		
		if( rk + 1 .le.  k - 1) then
          		do 101 i = rk + 1, k - 1  
c  Calculate the fist row in column i 
		           ri = i + 1 + idiag( i - 1 ) - idiag( i)
c  Calculate where the overlap begins  
			   j = MAX(ri,rk)
			   iloci = idiag( i) - (i-j)
			   ilock = idiag( k) - (k-j)
			   len = i - j
			   p = dotX(A(iloci),A(ilock),len)
c  Calculate yi in column k  
			   iloc = idiag( k) - (k - i)
			   A(iloc) = A(iloc) - p
101			continue
		end if
c  Calculate dk  
		if( rk .le. k - 1) then
          		do 102 i = rk , k - 1 
c  Calculate contribution to u(transpose)Du  
			   iloc = idiag( k) - (k - i)
			   t = A(iloc)
			   A(iloc) = t/A(idiag( i))
			   A(idiag( k)) = A(idiag( k)) - t*A(iloc)
102			continue
		end if
100     continue
	return
	end

c  This program performs back substitution on a skylined linear
c  system 

	subroutine solve( A, f, idiag, neq)

	include "tsconst.hf"

        integer i,j, k, iloci, ri, rj, ipntr, len
	double precision p, dotX

        integer idiag(mxdof), neq
	double precision f(mxdof), A(mxkmat)

c  forward substitution 

        do 100 i = 2, neq  
c  Calculate the fist row in column k 
		ri = i + 1 + idiag( i - 1 ) - idiag( i)
		if( ri .le. i - 1) then
			iloci = idiag( i) - (i - ri)
			len = i - ri
			p = dotX(A(iloci),f(ri),len)
			f(i) = f(i) - p
		end if
100     continue
c  diagonal scaling 

        do 200 i = 1, neq  
		if( A(idiag( i)) .ne. 0) then
			f(i) = f(i)/A(idiag( i))
		end if
200     continue
c  Backward substitution 
        do 300 j = neq, 2, -1 
c  Calculate the fist row in column j 
		rj = j + 1 + idiag( j - 1 ) - idiag( j)
		if( rj .le. j - 1) then
			ipntr = idiag( j) - j
        		do 301 i = rj , j - 1   
				f(i) = f(i) - f(j)*A(ipntr + i)
301			continue
		end if
300     continue
	return
	end


c  This subroutine takes the dot product of 2 vectors 
c
c		Updated 11/19/98
c

        function dotX(A,B,p)

	integer i,p
	double precision dotX,A(p),B(p)

        dotX=0.0
        do 100 i = 1, p   
              dotX = dotX + A(i)*B(i)
100	continue
        return
	end

        subroutine bzero1(x, n)

c  This subroutine zeroes out a 1-D matrix
c
c               Updated 11/19/98
c

        integer i, j, n
        double precision x(n)

        do 100 i = 1, n
              x(i) = 0.0
100     continue
        return
        end

        subroutine bzero2(x, n, m)

c  This subroutine zeroes out a 2-D matrix
c
c               Updated 11/19/98
c

        integer i, j, n, m
        double precision x(n,m)

        do 100 i = 1, n
           do 101 j = 1, m
              x(i,j) = 0.0
101        continue
100     continue
        return
        end
