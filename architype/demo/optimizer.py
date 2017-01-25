""" This class will optimize an individual mesh generated
by the grammar by iteratively calculating the required section 
sizes for optimal sizing, and then re-analysing the structure.  
Copyright (c) 2010
Michael Fenton
Hereby licensed under the GNU GPL v3."""

import analyser as AZR

class Optimizer():
    def __init__(self,uid,program):
        self.uid = uid  
        self.program = program
        self.analyser = AZR.Analyser(uid,program)
        self.bridge_weight=0
        self.fixed_list=[]
        self.load_nodes=[]
        self.nodeselfloads=[]
        self.load_elems=[]
        self.beams=[]
        self.stress_log=[]
        self.iterations = 3   
    
    def first_reassign_size(self,num,file_name):
        self.bridge_weight=0
        self.analyser.create_graph()
        self.analyser.apply_stresses()
        self.analyser.create_slf_file()
  #      self.copy_slf_file(num,file_name)
        self.analyser.test_slf_file()
        self.analyser.parse_results()
        stress_list = self.analyser.parse_results()
  #      self.log_results(stress_list)
        self.reassign_materials_quickly(stress_list)
      
    def reassign_size(self,num,file_name):
        self.bridge_weight=0
        self.analyser.create_slf_file()
    #    self.copy_slf_file(num,file_name)
        self.analyser.test_slf_file()
        self.analyser.parse_results()
        stress_list = self.analyser.parse_results()
    #    self.log_results(stress_list)
        self.reassign_materials_quickly(stress_list)

    def copy_slf_file(self,num,file_name):
        mesh= open(self.analyser.name,'r')
        kesh= open("slf_copies/slf_copy."+str(file_name)+"."+str(num),"w")
        for line in mesh:
            kesh.write(str(line))
        kesh.close()
        mesh.close()

    def find_weight(self):
        for i,beam in enumerate(self.analyser.edge_list):
            length = float(self.analyser.edge_list[i]['length'])
            zero = self.analyser.edge_list[i]['material']
            weight = length * float(self.analyser.beams[int(zero)]['unitweight']) # answer is in kg
            self.bridge_weight = float(self.bridge_weight) + float(weight)
        return self.bridge_weight

    def optimize_size(self,file_name,button=False):
        print "Optimizing member sizings for individual: ", file_name
      #  self.write_log(file_name)
        self.first_reassign_size(0,file_name)
        for i in range(self.iterations):
            n = i+1
            self.reassign_size(n,file_name)
     #   self.write_log(file_name)
        if button == True:
            self.analyser.show_analysis()      

    def write_log(self,name):
        log = file("stressLogs/stress_log."+str(name),"w")
        log.write("Stress reduction log for size optimization of individual " + str(name))
        for i,stress in enumerate(self.stress_log):
            log.write("\nRun "+str(i)+":\n")
            log.write("\tmax xx stress is:" + str(stress[0]))
            log.write("\n\tmax xy stress is:" + str(stress[1]))
            log.write("\n\tmax zx stress is:" + str(stress[2]))
            log.write("\n\tthe bridge weight (in kg) is:" + str(stress[3]))
        log.close()
    
    def log_results(self,stress_list):        
        for stress in stress_list:
            i,xx,xy,zx = stress['id'],stress['xx'],stress['xy'],stress['zx']
            stx, sty, stz = (0,0,0)
            if abs(xx) > stx:
                stx = abs(xx)
            if abs(xy) > sty:
                sty = abs(xy)
            if abs(zx) > stz:
                stz = abs(zx)
        weight = self.find_weight()
        big_log = [stx,sty,stz,weight]
        self.stress_log.append(big_log)       
        
    def reassign_materials(self,stress_list):   
        for stress in stress_list[::2]:
            i,xx,xy,zx = stress['id'],stress['xx'],stress['xy'],stress['zx']
            st = self.analyser.material['allowed_xx_compression']
            if abs(xx) < st and abs(xy) < st and abs(zx) < st:
                if int(self.analyser.edge_list[i]['material']) < (len(self.analyser.beams)-1):
                    self.analyser.edge_list[i]['material'] = int(self.analyser.edge_list[i]['material']) -1 
                    self.analyser.edge_list[i]['mass'] = float(self.analyser.edge_list[i]['length']) * float(self.analyser.beams[(int(self.analyser.edge_list[i]['material'])-1)]['unitweight'])*10
                # this means that the beam is over-performing, i.e. it is bigger than it needs to be and we can reduce its size.
            else:
                if int(self.analyser.edge_list[i]['material']) > 0:
                    self.analyser.edge_list[i]['material'] = int(self.analyser.edge_list[i]['material']) +1                               
                    self.analyser.edge_list[i]['mass'] = float(self.analyser.edge_list[i]['length']) * float(self.analyser.beams[(int(self.analyser.edge_list[i]['material'])-1)]['unitweight'])*10
                # this means that the beam is over-stressed and we have to make it bigger. it might be easiest to start off with ridiculously
                # massive beams and then gradually make everything smaller, rather than starting off small and making things bigger.

    def reassign_materials_quickly(self,stress_list):   
        for stress in stress_list[::2]:
            i,xx,xy,zx = stress['id'],stress['xx'],stress['xy'],stress['zx']
            st = self.analyser.material['allowed_xx_compression']
            governing_stress = abs(max(xx, xy, zx))
#            print '\nstress id:',i
#            print 'beam id:',self.analyser.edge_list[i]['id']
#            print 'governing stress:',governing_stress
            original_area = float(self.analyser.beams[int(self.analyser.edge_list[i]['material'])]['area'])
#            print 'original area:',original_area
#            print 'original member:',self.analyser.edge_list[i]['material']
            force = float(governing_stress)*original_area
            required_area = force/st
#            print "required_area:",required_area
#            print len(self.analyser.beams)
            if required_area > self.analyser.beams[155]['area']:
                self.analyser.edge_list[i]['material'] = 156
                self.analyser.edge_list[i]['mass'] = float(self.analyser.edge_list[i]['length']) * float(self.analyser.beams[155]['unitweight']) # answer is in Newtons
            if required_area < self.analyser.beams[0]['area']:
                self.analyser.edge_list[i]['material'] = 0
                self.analyser.edge_list[i]['mass'] = float(self.analyser.edge_list[i]['length']) * float(self.analyser.beams[0]['unitweight']) # answer is in Newtons
            for a,beam in enumerate(self.analyser.beams):
                if a > 0:
                    if self.analyser.beams[a-1]['area'] < required_area < self.analyser.beams[a]['area']:
#                        print "target area:", self.analyser.beams[a]['area']
#                        print 'target member:',self.analyser.beams[a]['id']
                        self.analyser.edge_list[i]['material'] = a
                        self.analyser.edge_list[i]['mass'] = float(self.analyser.edge_list[i]['length']) * float(self.analyser.beams[(int(self.analyser.edge_list[i]['material'])-1)]['unitweight']) # answer is in Newtons
 #                       print "new member:",self.analyser.edge_list[i]['material']
                        break
