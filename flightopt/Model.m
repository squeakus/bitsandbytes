%% Model: 
%A simulation of a multi line scan that calculates the shadow size
%% Input: 
% None
%% Returns: 
% Shadow size integer?
%%
clear all
close all
clc
% the buildings data:
Num=4;   % Number of buildings
% read the buidings vertices coordinates
points=xlsread('input3.xlsx','Sheet1');
mx=min(points(:,1));my=min(points(:,2));
maxx=max(points(:,1));maxy=max(points(:,2));
% the centroid of the scanned area
xcent=((maxx-mx)/2)
ycent=((maxy-my)/2)
xss=points(:,1);
yss=points(:,2);
zss=points(:,3);

% read the buildings topolog
faces=xlsread('input3.xlsx','Sheet2');
% the flight lines:
% angle with the positive x axis
% the angle for the perpindicular lines
theta1=120;
theta2=30;
fl_ht=30;

% distance for the center line from the centroid
d=0;                                                                        
% number of line in each direction
ng=5;
% spacing between scan lines
sg=10;                                                                       
[ag1 bg1 cg1]=grid_par(xcent,ycent,theta1,d,ng,sg);
[ag2 bg2 cg2]=grid_par(xcent,ycent,theta2,d,ng,sg);
zh=fl_ht*ones(ng,1);
lines=[ag1' bg1' cg1' zh;ag2' bg2' cg2' zh];
lines_size=size(lines);
tem_line=lines_size(1);
for linnes=1:tem_line
    points=[xss yss zss];
    points_cord=NaN(15,8,Num); % create NaN array with the expected size for the points array
    faces_ind=NaN(10,8,Num);  % create NaN array with the expected size for the faces indeces
    al=lines(linnes,1);bl=lines(linnes,2);cl=lines(linnes,3);fh=lines(linnes,4);
    counter=1;
    for i=1:15:length(points)
        temp=i+14;
        if temp>length(points)
            lim=length(points);
        else
            lim=i+14;
        end
        lim2=lim-i+1;
        zground=min(points((i:i+lim2-1),3));                                      
        for j=i:(i+lim2-1)
            % to find the ground level of a building
            [xl yl]=lpoint(al,bl,cl,points(j,1),points(j,2));
            % to add the light source for each point
            points(j,4:5)=[xl yl];
            if points(j,3)==0
                points(j,6:8)=[points(j,1) points(j,2) points(j,3)];
            else
                % to add the ground projection for each point.
                [xg yg zg]=mproject(xl,yl,fh,points(j,1),points(j,2),points(j,3),0,0,1,-zground);  
                if isnan(zg)~=1
                    zg=zground;
                end
                points(j,6:8)=[xg yg zg];
            end
        end
    end
    for i=1:15:length(points)
        temp=i+14;
        if temp>length(points)
            lim=length(points);
        else
            lim=i+14;
        end
        lim2=lim-i+1;
        % for each building, the points are stored [ point numbe XYZ Building number]
        points_cord(1:lim2,:,counter)=points(i:lim,:);                            
        counter=counter+1;
    end
    % to find the min level for each bulding:
    zz=[];
    for i=1:Num
        zz(i)=min(points_cord(:,3,i));
    end
    counter=1;
    for j=1:10:length(faces)
        temp=j+9;
        if temp>length(faces)
            lim=length(faces);
        else
            lim=j+9;
        end
        sub=faces(j:lim,:);
        lim2=lim-j+1;
        for jj=1:lim2
            sub_sub=sub(jj,:);
            % for each building, the faces are stored as the index of the corners cordinates
            faces_ind(jj,1:length(sub_sub),counter)=sub_sub;                         
        end
        counter=counter+1;
    end
    %% Processing
    % to find the convex hull for each building.
    k=NaN(Num,10);
    kf=NaN(Num,10);
    for i=1:Num
        ktem=outerbound(points_cord(:,:,i),fh);
        k(i,1:length(ktem))=ktem';
        kftem=footprint([ points_cord(:,1,i) points_cord(:,2,i)]);
        % to find the indeces for the foot print
        kf(i,1:length(kftem))=kftem';                                             
    end
    % to classify the points into visible and none visible
    vis_ind=NaN(Num,15);
    for i=1:Num
        tem=points_cord(:,1,i);
        tem(isnan(tem))=[];
        tot_ind_tem=1:length(tem);
        ktem=k(i,:);
        ktem(isnan(ktem))=[];
        tot_ind=ismember(tot_ind_tem,ktem);
        xbtem=points_cord(ktem,1,i);
        ybtem=points_cord(ktem,2,i);
        counter=1;
        vis_indt=[];
        for j=1:length(tot_ind)
            chk=tot_ind(j);
            if chk==0
                [inn onn]=inpolygon(points_cord(j,1,i),points_cord(j,2,i),xbtem,ybtem);
                if inn==1 && onn==0
                    vis_indt(counter)=2;
                elseif inn==1 && onn==1
                    ztem=zval(points_cord(j,1,i),points_cord(j,2,i),points_cord(ktem,1,i),points_cord(ktem,2,i),points_cord(ktem,3,i));
                    if ztem<points_cord(j,3,i)
                        vis_indt(counter)=2    ;
                    else
                        vis_indt(counter)=-1;
                    end
                else
                    vis_indt(counter)=-2;
                end
            elseif chk==1
                vis_indt(counter)=1;
            end
            counter=counter+1;
        end
        vis_ind(i,1:length(vis_indt))=vis_indt;
    end
    % to classify the faces into visible and none visible
    for i=1:Num
        test=faces_ind(:,1,i);
        test(isnan(test))=[];
        len_tem=length(test);
        for j=1:len_tem
            ind5=faces_ind(j,:,i);
            ind5(isnan(ind5))=[];
            vis_fc=vis_ind(i,ind5);
            ch1=length(find(vis_fc==-1));
            ch2=length(find(vis_fc==-2));
            if ch1>0
                faces_ind(j,8,i)=-1;
            elseif ch2>0
                faces_ind(j,8,i)=-2;
            end
        end
    end
    % to find the buildings projecting shadow on the building
    bldg_ocldr=NaN(Num,1);
    % figure(1)
    for i=1:Num
        ind=kf(i,:);
        ind(isnan(ind))=[];
        xb=points_cord(ind,1,i);
        yb=points_cord(ind,2,i);
        [xbc ybc]=poly2cw(xb,yb);
        %     plot(xb,yb,'b')
        %     hold all
        %     name=num2str(i);
        %     text(mean(xb),mean(yb),name)
        %     hold all
        counter=1;
        for j=1:Num
            %        innnddd=[i j]
            ind2=k(j,:);
            ind2(isnan(ind2))=[];
            %        xbch(j,1:length(ind2),i)=points_cord(ind2,6,j)
            %        ybch(j,1:length(ind2),i)=points_cord(ind2,7,j)
            xbch=points_cord(ind2,6,j);
            ybch=points_cord(ind2,7,j);
            [xbchc ybchc]=poly2cw(xbch,ybch);
            [xtem ytem]=polybool('and',xbc,ybc,xbchc,ybchc);
            areatem=polyarea(xtem,ytem);
            if length(temp)~=0 && i~=j  && zz(i)<=zz(j) && areatem>2
                bldg_ocldr(i,counter)=j;
                counter=counter+1;
            end
            %        plot(xbch,ybch,'r')
            %        hold all
            %        name=num2str(j);
            %        text(mean(xbch),mean(ybch),name)
            %        hold all
        end
    end
    % to project the shadows.
     % size_tem(2) is the maximum number of posible shadows to be projected on the surface
    size_tem=size(bldg_ocldr);
    globshd=NaN(Num*size_tem(2)*10,15,3);
    % nnormal=NaN(Num*size_tem(2)*10,4);
    counter=1;
    % figure(2)
    % size_tem(1)
    for i=1:Num
        for j=1:10
            ind3=faces_ind(j,1:7,i);
            ind3(isnan(ind3))=[];
            xfcb=points_cord(ind3,1,i); % faces coordinates
            yfcb=points_cord(ind3,2,i);
            zfcb=points_cord(ind3,3,i);
            %        fill3(xfcb,yfcb,zfcb,'y','FaceAlpha', 0.05)
            %        hold all
            for m=1:size_tem(2)
                globshd(counter,1,1)=i;
                globshd(counter,2,1)=j;
                globshd(counter,1,2)=i;
                globshd(counter,2,2)=j;
                globshd(counter,1,3)=i;
                globshd(counter,2,3)=j;
                if length(ind3)~=0 && faces_ind(j,8,i)~=-2 && faces_ind(j,8,i)~=-1
                    xo=xfcb(1);x1=xfcb(2);x2=xfcb(3);yo=yfcb(1);y1=yfcb(2);y2=yfcb(3);
                    zo=zfcb(1);z1=zfcb(2);z2=zfcb(3);
                    [a b c d]=plan3p(xo,yo,zo,x1,y1,z1,x2,y2,z2);
                    if bldg_ocldr(i,m)~=0 && isnan(bldg_ocldr(i,m))~=1
                        ind4=k(bldg_ocldr(i,m),:);
                        ind4(isnan(ind4))=[];
                        xshad=points_cord(ind4,1,bldg_ocldr(i,m));
                        yshad=points_cord(ind4,2,bldg_ocldr(i,m));
                        zshad=points_cord(ind4,3,bldg_ocldr(i,m));
                        xlight=points_cord(ind4,4,bldg_ocldr(i,m));
                        ylight=points_cord(ind4,5,bldg_ocldr(i,m));
                        zlight=fh*ones(length(xlight),1);
                        [xfcshd yfcshd zfcshd]=plnprojfcd(xshad,yshad,zshad,xlight,ylight,zlight,a,b,c,d);
                        xfcshd(isnan(xfcshd))=[];yfcshd(isnan(yfcshd))=[];zfcshd(isnan(zfcshd))=[];
                        [xintshd yintshd zintshd]=polybol3d(xfcb,yfcb,zfcb,xfcshd,yfcshd,zfcshd,a,b,c,d);
                        %          fill3(xintshd,yintshd,zintshd,linnes,'FaceAlpha', 0.3)
                        %         hold all
                        globshd(counter,3:length(xintshd)+2,1)=xintshd;
                        globshd(counter,3:length(xintshd)+2,2)=yintshd;
                        globshd(counter,3:length(xintshd)+2,3)=zintshd;
                        %                nnormal(counter,:)=[a b c d];
                    end
                end
                counter=counter+1;
            end
        end
    end
    % to find the common walls:
    counter=1;
    comwls=NaN(Num*10,12,3);
    for i=1:Num
        for j=1:10
            if isnan(faces_ind(j,1,i))==0
                fc_ind=faces_ind(j,1:7,i);
                fc_ind(isnan(fc_ind))=[];
                xfctem=points_cord(fc_ind,1,i);
                yfctem=points_cord(fc_ind,2,i);
                zfctem=points_cord(fc_ind,3,i);
                p11=[xfctem(1) yfctem(1) zfctem(1)];
                p12=[xfctem(2) yfctem(2) zfctem(2)];
                p13=[xfctem(3) yfctem(3) zfctem(3)];
                for ii=1:Num
                    for jj=1:10
                        if isnan(faces_ind(jj,1,ii))==0
                            fc_ind_tem=faces_ind(jj,1:7,ii);
                            fc_ind_tem(isnan(fc_ind_tem))=[];
                            xfctem1=points_cord(fc_ind_tem,1,ii);
                            yfctem1=points_cord(fc_ind_tem,2,ii);
                            zfctem1=points_cord(fc_ind_tem,3,ii);
                            p21=[xfctem1(1) yfctem1(1) zfctem1(1)];
                            p22=[xfctem1(2) yfctem1(2) zfctem1(2)];
                            p23=[xfctem1(3) yfctem1(3) zfctem1(3)];
                            [simpl a b c d]=checksimp(p11,p12,p13,p21,p22,p23);
                            if simpl==1 && i~=ii && j~=jj
                                [XXX YYY ZZZ]=polybol3d(xfctem,yfctem,zfctem,xfctem1,yfctem1,zfctem1,a,b,c,d);
                                comwls(counter,1:length(XXX),1)=XXX;
                                comwls(counter,1:length(XXX),2)=YYY;
                                comwls(counter,1:length(XXX),3)=ZZZ;
                                %                 fill3(XXX,YYY,ZZZ,'b')
                                %                 hold all
                            end
                        end
                    end
                end
            end
            counter=counter+1;
        end
    end
    counter=1;
    siz_tem=size(faces_ind);
    normals=NaN(Num*siz_tem(1),4);
    for i=1:Num
        for j=1:siz_tem(1)
            if isnan(faces_ind(j,1,i))==0
                xo=points_cord(faces_ind(j,1,i),1,i);yo=points_cord(faces_ind(j,1,i),2,i);zo=points_cord(faces_ind(j,1,i),3,i);
                x1=points_cord(faces_ind(j,2,i),1,i);y1=points_cord(faces_ind(j,2,i),2,i);z1=points_cord(faces_ind(j,2,i),3,i);
                x2=points_cord(faces_ind(j,3,i),1,i);y2=points_cord(faces_ind(j,3,i),2,i);z2=points_cord(faces_ind(j,3,i),3,i);
                
                [a b c d]=plan3p(xo,yo,zo,x1,y1,z1,x2,y2,z2);
            else
                a=NaN;b=NaN;c=NaN;d=NaN;
            end
            normals(counter,:)=[a b c d];
            counter=counter+1;
        end
    end
    
    % to join the shadows on one face.
    counter=1;
    m;
    tshdtem=NaN(Num*10,m*20,3);
    tshdgtem=NaN(Num*10,m*20,2);
    for i=1:m*10:(Num*m*10)
        for j=i:m:(i-1+(10*m))
            sub_shdx(1:m,:)=globshd(j:(j+m-1),3:end,1);
            sub_shdy(1:m,:)=globshd(j:(j+m-1),3:end,2);
            sub_shdz(1:m,:)=globshd(j:(j+m-1),3:end,3);
            a=normals(counter,1);b=normals(counter,2);c=normals(counter,3);d=normals(counter,4);
            [xtem ytem ztem xgtem ygtem]=oneplnoneln(sub_shdx,sub_shdy,sub_shdz,a,b,c,d);
            tshdtem(counter,1:length(xtem),1)=xtem;tshdtem(counter,1:length(xtem),2)=ytem;tshdtem(counter,1:length(xtem),3)=ztem;
            tshdgtem(counter,1:length(xtem),1)=xgtem;tshdgtem(counter,1:length(xtem),2)=ygtem;
            counter=counter+1;
            %    plot3(xtem,ytem,ztem)
            %    hold all
        end
    end
    size(tshdtem);
    linnes;
    eval([['all_shadows' num2str(linnes)],'=tshdtem;'])
    eval([['all_shadowsg' num2str(linnes)],'=tshdgtem;'])
    eval([['psbdhs' num2str(linnes)],'=size_tem;'])
    eval([['normalss' num2str(linnes)],'=normals;'])
    
    clearvars -except xss yss zss  faces lines lines_size tem_line Num all_shadows* psbdhs* normalss1 dd
end
% figure(3)
countter=1;
for i=1:length(normalss1)
    a=normalss1(i,1);b=normalss1(i,2);c=normalss1(i,3);d=normalss1(i,4);
    pl2x=[];pl2y=[];
    for k=1:tem_line
        shd1=eval(['all_shadowsg' num2str(k)]) ;
        pl1x=shd1(i,:,1);pl1x(isnan(pl1x))=[];
        pl1y=shd1(i,:,2);pl1y(isnan(pl1y))=[];
        pl2x;
        pl2y;
        [X Y]=polybool('union',pl1x,pl1y,pl2x,pl2y);
        pl2x=X;pl2y=Y;
    end
    if isempty(pl2x)~=1
        %         pl2x
        %         pl2y
        Xt=[];Yt=[];Zt=[];
        for j=1:length(pl2x)
            [xtem ytem ztem]=mproject(100000,100000,100000,pl2x(j),pl2y(j),0,a,b,c,d);
            xtem=floor(xtem*10000)/10000;
            ytem=floor(ytem*10000)/10000;ztem=floor(ztem*10000)/10000;
            Xt(j)=xtem;Yt(j)=ytem;Zt(j)=ztem;
        end
        % plot3(Xt,Yt,Zt)
        % hold all
        Xt;
        Yt;
        Zt;
        
        coun=1;
        ind1=isnan(Xt);
        lits=[];
        for ii=1:length(Xt)
            if ind1(ii)==1
                lits(coun)=ii;
                coun=coun+1;
            end
        end
        if isempty(lits)==1
            %     fill3(Xt,Yt,Zt,'c')
            %     hold all
            %     a
            %     b
            %     c
            aarrea(countter)=areea2(Xt,Yt,Zt,a,b,c);
            countter=countter+1;
            
        else
            gg=[0 lits];
            
            for iii=1:length(gg)
                if iii==length(gg)
                    sub_x=Xt(gg(iii)+1:end);
                    sub_y=Yt(gg(iii)+1:end);
                    sub_z=Zt(gg(iii)+1:end);
                    % fill3(sub_x,sub_y,sub_z,'b')
                    aarrea(countter)=areea2(sub_x,sub_y,sub_z,a,b,c);
                    % hold all
                    countter=countter+1;
                else
                    sub_x=Xt(gg(iii)+1:gg(iii+1)-1);
                    sub_y=Yt(gg(iii)+1:gg(iii+1)-1);
                    sub_z=Zt(gg(iii)+1:gg(iii+1)-1);
                    % fill3(sub_x,sub_y,sub_z,'b')
                    % hold all
                    aarrea(countter)=areea2(sub_x,sub_y,sub_z,a,b,c);
                    countter=countter+1;
                end
            end
        end
    end
end
