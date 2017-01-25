"""Creates an evolver objects and manages GUI
Copyright (c) 2010 Jonathan Byrne, Erik Hemberg and James McDermott
Hereby licensed under the GNU GPL v3."""
import sys, os, re, subprocess, time, evolver, grammar,tkFileDialog
from PIL import Image, ImageTk
import Tkinter as TK
import analyser as AZR

class GE(object):
    """Creates evolver instance and initialise first generation"""

    def __init__(self):
        self.generation = 0
        evolver.TIME = time.time()
        evolver.SAVE_BEST = True
        evolver.CODON_SIZE = 100
        evolver.ELITE_SIZE = 1
        evolver.POPULATION_SIZE = 35
        evolver.GENERATION_SIZE = 35
        evolver.FRONT_FOLDER = "frontData"
        evolver.GENERATIONS = 5
        evolver.DEFAULT_FIT = 100000000000000
        evolver.MUTATION_PROBABILITY = 0.015
        evolver.CROSSOVER_PROBABILITY = 0.7
        evolver.GRAMMAR_FILE = "grammars/jon_pylon10.bnf"
        evolver.FITNESS_FUNCTION = evolver.StructuralFitness()
        evolver.IMG_COUNTER = 0

        self.pop_size = evolver.POPULATION_SIZE
        self.grammar = grammar.Grammar(evolver.GRAMMAR_FILE)
        self.individuals = evolver.initialise_population(evolver.POPULATION_SIZE)
        for idx, indiv in enumerate(self.individuals):
            indiv.uid = idx
        self.selection = lambda x: evolver.tournament_selection(x, evolver.POPULATION_SIZE)
        evolver.evaluate_fitness(self.individuals, self.grammar,
                                 evolver.FITNESS_FUNCTION)
        self.best_ever = min(self.individuals)
        self.fronts = []
        self.individuals.sort()
        
        print "creating meshes"
        evolver.create_meshes(self.individuals)
        evolver.print_stats(1, self.individuals)

    def step(self):
        """creates next generation"""
        print("no. of indivs: " + str(len(self.individuals)))
        self.individuals, self.fronts, self.best_ever = evolver.step(
            self.individuals, self.fronts, self.grammar,
            self.selection, evolver.FITNESS_FUNCTION, self.best_ever)
        evolver.print_stats(self.generation, self.individuals)
        sys.stderr.write("Gen: " + str(self.generation) + "\n")
        self.generation += 1


class AutoScrollbar(TK.Scrollbar):
    """a scrollbar that hides itself if it's not needed.  only
    works if you use the grid geometry manager."""

    def set(self, low, high):
        if float(low) <= 0.0 and float(high) >= 1.0:
            # grid_remove is currently missing from Tkinter!
            self.tk.call("grid", "remove", self)
        else:
            self.grid()
        TK.Scrollbar.set(self, low, high)

    def pack(self, **kw):
        raise TclError, "cannot use pack with this widget"

    def place(self, **kw):
        raise TclError, "cannot use place with this widget"


class GUI:
    """Responsible for managing evolver object and generating buttons"""

    def __init__(self, root):
        self.root = root
        self.defCol = root.cget("bg")
        self.images = {}
        self.start_time = time.time()
        self.buttons = {}
        self.chosen = 0
        self.int_muts = 0
        self.struct_muts = 0
        self.width = 7
        self.last_button = None
        self.default_fit = 10000000000000000
        self.rw, self.col = 1, 0
        self.rw, self.col = 1, 0
        self.images = {}
        self.buttons = {}
        self.ppm_list = []
        #defining dictionaries
        self.muts = {'int': 0, 'nodal': 0, 'struct': 0}
        self.size = {'frameW': 1100, 'frameH': 690, 'imgW': 140,
                     'imgH': 140}

        #set up folders and specify OS
        self.dxf_folder = os.getcwd()+'/dxf/'
        self.pop_folder = os.getcwd() + '/population/'
        self.save_folder = os.getcwd() + '/saved/'
        if sys.platform == 'darwin':
            print "using medit for osx"
            self.meditcmd = self.pop_folder + "macMedit "
            self.showcmd = self.meditcmd

        elif sys.platform == 'linux2':
            print "using medit for linux"
            self.meditcmd = self.pop_folder + "linuxMedit "
            self.showcmd = self.pop_folder + "linuxShow "

        # wipe last population and start evolver
        if len(os.listdir(self.pop_folder)) > 2:
            self.clear_folder()
        self.ge = GE()
        self.create_ppm()

        #make the scrollbars
        self.vscrollbar = AutoScrollbar(root, width=20)
        self.vscrollbar.grid(row=1, column=1, sticky=TK.N + TK.S)
        self.hscrollbar = AutoScrollbar(root, width=20,
                                        orient=TK.HORIZONTAL)
        self.hscrollbar.grid(row=2, column=0, sticky=TK.E + TK.W)

        #creating a control panel
        self.info_frame = TK.Frame(root)
        self.info_frame.rowconfigure(1, weight=1)
        self.info_frame.grid(row=0)
        #create info label
        self.label = TK.Label(self.info_frame, font=("Helvetica", 16))
        self.info_frame.update_idletasks()
        self.label.grid(row=0, columnspan=self.width)
        self.label['text'] = "starting architype"

        #create canvas
        self.canvas = TK.Canvas(root, height=self.size['frameH'],
                                width=self.size['frameW'],
                                yscrollcommand=self.vscrollbar.set,
                                xscrollcommand=self.hscrollbar.set)
        self.canvas.grid(row=1, column=0,
                         sticky=TK.N + TK.S + TK.E + TK.W)
        self.vscrollbar.config(command=self.canvas.yview)
        self.hscrollbar.config(command=self.canvas.xview)

        # make the canvas expandable
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # create canvas contents
        self.mainframe = TK.Frame(self.canvas)
        self.mainframe.rowconfigure(1, weight=1)
        self.mainframe.columnconfigure(1, weight=1)
        text = ("Please select the bridges you want to evolve")
        self.create_buttons(self.pop_folder, text)

        # create the popup menu
        self.popup = TK.Menu(root, tearoff=0)
        self.popup.add_command(label="View (v)",
                               command=lambda: self.show_indiv())
        self.popup.add_command(label="Analyse (a)",
                               command=lambda: self.analyse_indiv())
        self.popup.add_command(label="Show Fitness (f)",
                               command=lambda: self.show_fitness())                      
        self.popup.add_separator()
        self.popup.add_command(label="Small Mutate (s)",
                               command=lambda: self.mutate("nodal"))
        self.popup.add_command(label="Medium Mutate (m)",
                               command=lambda: self.mutate("int"))
        self.popup.add_command(label="big Mutate (b)",
                               command=lambda: self.mutate("big"))
        self.popup.add_command(label="Evolve (enter/space)",
                               command=lambda: self.next_generation())
        self.popup.add_command(label="Optimize Size (o)",
                               command=lambda:self.optimize())
        self.popup.add_separator()
        self.popup.add_command(label="Write Individual to file",
                               command=lambda: self.save_indiv())
        self.popup.add_command(label="Save population to file",
                               command=lambda: self.save_population())

        self.popup.add_command(label="Load Population from file",
                               command=lambda: self.load_pop())
        self.popup.add_command(label="Save as DXF (d)",
                               command=lambda:self.save_dxf())
        self.popup.add_separator()
        self.popup.add_command(label="show all selected in run",
                               command=lambda: self.show_best())

        #add listener for frame updates
        self.canvas.create_window(0, 0, anchor=TK.NW, window=self.mainframe)
        self.mainframe.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

        #bind some keys
        root.bind("<Return>", lambda event: self.next_generation())
        root.bind("<space>", lambda event: self.next_generation())
        root.bind("<Up>", lambda event: self.move_vert(event, -1))
        root.bind("<Down>", lambda event: self.move_vert(event, 1))
        root.bind("<Left>", lambda event: self.move_horiz(event, -1))
        root.bind("<Right>", lambda event: self.move_horiz(event, 1))
        root.bind("<a>", lambda event: self.analyse_indiv())
        root.bind("<f>", lambda event: self.show_fitness())
        root.bind("<v>", lambda event: self.show_indiv())
        root.bind("<o>", lambda event:self.optimize())
        root.bind("<d>", lambda event:self.save_dxf())
        root.bind("<s>", lambda event: self.mutate("nodal"))
        root.bind("<m>", lambda event: self.mutate("int"))
        root.bind("<b>", lambda event: self.mutate("struct"))

    def mutate(self, mut_op):
        """record mutation type and mutate individual """
        self.show_msg("mutated individual " + self.last_button
                      + " with " + mut_op + " mutation")
        if mut_op == "int":
            self.muts['int'] += 1
        if mut_op == "nodal":
            self.muts['nodal'] += 1
        if mut_op == "struct":
            self.muts['struct'] += 1

        for indiv in self.ge.individuals:
            if indiv.uid == int(self.last_button):
                self.clear_indiv(indiv.uid)
                indiv = evolver.mutate_individual(indiv, self.ge.grammar,
                                                  mut_op)
        self.update_image(self.last_button)

    def update_image(self, uid):
        """refresh ppm after a mutation event"""
        full_ppm_name = self.pop_folder + "indiv." + uid + ".ppm"
        full_mesh_name = self.pop_folder + "indiv." + uid + ".mesh"
        #linux medit has simplified command
        if sys.platform == 'linux2':
            cmd = self.meditcmd + " " + full_mesh_name
        else:
            cmd = self.meditcmd + "-xv 600 600 " + full_mesh_name
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE)
        process.communicate()
        tmp_img = Image.open(full_ppm_name)
        tmp_img = tmp_img.resize((self.size['imgW'], self.size['imgH']),
                                 Image.ANTIALIAS)
        self.images[uid] = ImageTk.PhotoImage(tmp_img)

        self.buttons[uid].configure(image=self.images[uid])

    def get_num(self, filename):
        return float(re.findall(r'\d+', filename)[0])

    def create_buttons(self, folder, text):
        """assigns images to the buttons and attaches listeners"""
        self.rw, self.col = 1, 0
        self.images = {}
        self.buttons = {}
        self.ppm_list = []
        self.label['text'] = text
        self.info_frame.update_idletasks()
        #self.label.grid(row=0, columnspan=self.width)
        for file_name in os.listdir(folder):
            if file_name.endswith('.ppm'):
                self.ppm_list.append(file_name)
        self.ppm_list.sort(lambda x, y: cmp(self.get_num(x), self.get_num(y)))
        for ppm_name in self.ppm_list:
            full_name = folder + ppm_name
            name = ppm_name.strip('.ppm')
            uid = name.strip('indiv.')
            tmp_img = Image.open(full_name)
            tmp_img = tmp_img.resize((self.size['imgW'], self.size['imgH']),
                                     Image.ANTIALIAS)
            self.images[uid] = ImageTk.PhotoImage(tmp_img)
            self.buttons[uid] = TK.Button(self.mainframe, command=lambda
                                        x=uid: self.button_handler(x),
                                        image=self.images[uid],
                                        bd=5)
            self.buttons[uid].grid(row=self.rw, column=self.col)
            self.buttons[uid].bind("<Button-3>", lambda event, x=uid:
                                       self.right_click(event, x))
            self.col += 1
            if self.col == self.width:
                self.col = 0
                self.rw += 1

    def next_generation(self):
        print "evolving next gen:"
        self.show_msg("creating next generation, please wait.(reticulating)",
                      "red")
        self.clear_folder()
        time.sleep(0.5)
        self.show_msg("creating next generation, please wait. (generating)",
                      "orange")
        self.ge.step()
        self.show_msg("creating next generation, please wait. (analysing)",
                      "green")
        self.create_ppm()
        text = ("Generation: " + str(self.ge.generation)
               + " Please select the bridges you want to evolve")
        self.build_frame(self.pop_folder, text)

    def build_frame(self, source, text):
        """rebuilds button frame when a new population is created"""
        self.mainframe.destroy()
        self.mainframe = TK.Frame(self.canvas)
        self.mainframe.rowconfigure(1, weight=1)
        self.mainframe.columnconfigure(1, weight=1)
        self.create_buttons(source, text)
        self.canvas.create_window(0, 0, anchor=TK.NW, window=self.mainframe)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def show_msg(self, message, color=None):
        """updates message at the top of the frame, it really should
        have a separate frame"""
        self.label['bg'] = color
        self.label['text'] = message
        self.info_frame.update_idletasks()

    def button_handler(self, name):
        """records selected individuals and assigns fitness values"""
        self.buttons[name].focus_force()
        self.last_button = name
        current_indiv = None
        for indiv in self.ge.individuals:
            if indiv.uid == int(name):
                print "found:", name
                current_indiv = indiv
        if self.buttons[name]['background'] == "green":
            print "unassigning fitness to indiv", name
            for indiv in self.ge.individuals:
                if indiv.uid == int(self.last_button):
                    analyser = AZR.Analyser(indiv.uid,indiv.phenotype,False)
                    current_indiv.fitness = analyser.test_mesh()
            print current_indiv.fitness
            self.buttons[name]['background'] = self.defCol
            self.buttons[name]['relief'] = "raised"
        else:
            print "assigning good fitness to indiv:", name
            current_indiv.fitness = [0, 0, 0]
            self.buttons[name]['background'] = "green"
            self.buttons[name]['relief'] = "sunken"
            self.save_indiv("best" + str(self.chosen))
            self.chosen += 1

    def right_click(self, event, name):
        print "button name", name
        self.last_button = name
        self.show_popup(event)

    def move_vert(self, event, val):
        """handler to allow scrolling with  arrow keys"""
        self.canvas.yview('scroll', val, 'units')

    def move_horiz(self, event, val):
        """handler to allow scrolling with  arrow keys"""
        self.canvas.xview('scroll', val, 'units')

    def show_popup(self, event):
        try:
            self.popup.tk_popup(event.x_root, event.y_root, 0)
        finally:
            self.popup.grab_release()

    def clear_indiv(self, uid):
        cmd = "rm " + self.pop_folder + "indiv." + self.last_button + ".*"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE)
        process.communicate()

    def clear_folder(self, folder=None):
        if folder == None:
            cmd = "rm " + self.pop_folder + "*.*"
        else:
            cmd = "rm " + folder
        fnull = open(os.devnull, 'w')
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE, stderr=fnull)
        fnull.close()
        process.communicate()

    def create_ppm(self):
        print "creating ppms"
        cmd = (self.meditcmd + self.pop_folder + "indiv -a 0 "
               + str(self.ge.pop_size))
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE)
        process.communicate()

    def show_indiv(self):
        """handler that opens a mesh using medit"""
        print "opening medit on individual", self.last_button
        mesh_name = self.pop_folder + "indiv." + self.last_button + ".mesh"
        cmd = self.showcmd + " " + mesh_name
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE)
        process.communicate()

    def show_fitness(self):
        """Prints the current fitness of an individual"""
        for indiv in self.ge.individuals:
            if indiv.uid == int(self.last_button):
                print "Compliance, displacement, weight:", indiv.fitness

    def analyse_indiv(self):
        """handler that shows stresses using slffea"""
        print "analysing individual:", self.last_button
        for indiv in self.ge.individuals:
            if indiv.uid == int(self.last_button):
                evolver.run_analysis(indiv)

    def optimize(self,name=None):
        for indiv in self.ge.individuals:
            if indiv.uid == int(self.last_button):
                if name == None:
                    file_name = "gen"+str(self.ge.generation)+"ind"+self.last_button
                else:
                    file_name = name
                evolver.run_optimization(indiv,file_name,button=True)
    
    def save_indiv(self, name=None):
        print "saving individual:", self.last_button
        for indiv in self.ge.individuals:
            if indiv.uid == int(self.last_button):
                if name == None:
                    file_name = (self.save_folder + "gen"
                                + str(self.ge.generation) + "ind"
                                + self.last_button + ".dat")
                else:
                    file_name = self.save_folder + name + ".dat"
                self.show_msg("saved bridge " + self.last_button + " in file "
                              + file_name)
                save_file = open(file_name, 'w')
                save_file.write("genome:" + str(indiv.genome) + "\n")
                save_file.write("phenotype:" + indiv.phenotype + "\n")
                save_file.close()

    def save_population(self):
        print "archi save pop"
        evolver.save_pop(self.ge.individuals)

    def save_dxf(self):
        print "saving individual as DXF:",self.last_button
        for indiv in self.ge.individuals:
            if indiv.uid == int(self.last_button):
                analyser = AZR.Analyser(indiv.uid,indiv.phenotype,False)
                analyser.create_graph()
                analyser.save_dxf(self.ge.generation, name='indiv') 
                self.show_msg("saved bridge as DXF "+self.last_button+" in dxf folder")

    def show_best(self, load_pop=True):
        """prints out mutation stats and time taken and then shows all
        the selections the user made"""
        self.clear_folder(self.save_folder + "*.mesh")
        self.clear_folder(self.save_folder + "*.ppm")

        print ("Finished Run. " + "int mutations: " + str(self.muts['int'])
               + " nodal mutations:" + str(self.muts['nodal']) +
               " struct mutations: " + str(self.muts['struct']))
        time_taken = time.time() - self.start_time
        print "time taken:", round(time_taken, 2), "seconds"

    def load_pop(self):
        results = []
        self.clear_folder()
        dat_file = tkFileDialog.askopenfile(parent=self.root,
                                            filetypes=[("dat file","*.dat")],
                                            initialdir=os.getcwd(), 
                                            mode='rb',title='Choose a file')
        #read results
        print "loading file", dat_file.name
        if dat_file != None:
            for line in dat_file:
                if not line.startswith('#'):
                    line = line.rstrip()
                    array = line.split(';')
                    result = {'uid': array[0],
                              'fitness': array[1],
                              'genome': array[2]}
                    results.append(result)
            dat_file.close()
       
        for result in results:
            uid = result['uid']
            genome = eval(result['genome'])
            evolver.build_individual(self.pop_folder + uid, genome,
                                     self.ge.grammar)
            full_mesh_name = self.pop_folder + uid + ".mesh"
            if sys.platform == 'linux2':
                cmd = self.meditcmd + " " + full_mesh_name
            else:
                cmd = self.meditcmd + "-xv 600 600 " + full_mesh_name
            process = subprocess.Popen(cmd, shell=True,
                                       stdout=subprocess.PIPE,
                                       stdin=subprocess.PIPE)
            process.communicate()
        text = "Showing saved bridges, to continue evolving just press enter"
        self.build_frame(self.pop_folder, text)


if __name__ == '__main__':
    ROOT = TK.Tk()
    ROOT.title("Architype: Evolutionary Architecture")
    ROOT.geometry('+0+0')
    MYGUI = GUI(ROOT)
    ROOT.mainloop()
    print "window closed"
    MYGUI.show_best(False)
