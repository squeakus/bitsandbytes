import sys, os, re, subprocess, time, datetime, evolver
from PIL import Image, ImageTk
import Tkinter as TK


class GE(object):

    def __init__(self):
        self.generation = 0
        evolver.TIME = time.time()
        evolver.SAVE_BEST = False
        evolver.CODON_SIZE = 100
        evolver.ELITE_SIZE = 0
        evolver.POP_SIZE = 200
        evolver.GENERATION_SIZE = 1
        evolver.GENERATIONS = 50
        evolver.DEFAULT_FIT = 100000
        evolver.MUTATION_PROBABILITY = 0.015
        evolver.CROSSOVER_PROBABILITY = 0.0
        evolver.GRAMMAR_FILE = "grammars/support2.bnf"
        evolver.FITNESS_FUNCTION = evolver.StructuralFitness()
        evolver.IMG_COUNTER = 0

        self.popSize = evolver.POP_SIZE
        self.grammar = evolver.Grammar(evolver.GRAMMAR_FILE)
        self.individuals = evolver.initialise_population(evolver.POP_SIZE)
        for idx, indiv in enumerate(self.individuals):
            indiv.UID = idx
            if idx > 0:
                indiv.genome = None
        evolver.evaluate_fitness(self.individuals, self.grammar,
                                 evolver.FITNESS_FUNCTION)
        self.best_ever = min(self.individuals)
        self.individuals.sort()
        evolver.create_meshes(self.individuals)
        evolver.print_stats(1, self.individuals)


class GUI:

    def __init__(self, root, target_name, test=False):
        self.background = root.cget("bg")
        if test:
            self.saveName = "test"
        self.finished = False
        now = datetime.datetime.now()
        self.timeStamp = (str(now.day) + "_" + str(now.month) + "_"
                          + str(now.hour) + str(now.minute) + str(now.second))
        self.mutOp = None
        self.nodal_images = []
        self.struct_images = []
        self.root = root
        self.current_image = None
        self.current_indiv = None
        self.current_rule = None
        self.info_label = None
        self.struct_counter = None
        self.nodal_counter = None
        self.prods = 0
        self.ppm_count = 0
        self.index = 0
        self.startTime = time.time()
        self.defaultFit = 100000
        #defining dictionaries
        self.muts = {'chosen': 0, 'nodalMuts': 0,
                    'intMuts': 0, 'structMuts': 0}
        self.size = {'frameW': 1000, 'frameH': 1000, 'imgW': 600,
                     'imgH': 600, 'targetW': 600, 'targetH': 600}
        #setting up folders and getting OS
        self.popFolder = os.getcwd() + '/population/'
        self.saveFolder = os.getcwd() + '/saved/'
        if sys.platform == 'darwin':
            print "using medit for osx"
            self.meditcmd = self.popFolder + "macMedit "
        elif sys.platform == 'linux2':
            print "using medit for linux"
            self.size = {'frameW': 700, 'frameH': 700, 'imgW': 400,
                         'imgH': 400, 'targetW': 400, 'targetH': 400}
            self.meditcmd = self.popFolder + "animedit "

        # wipe last population and start evolver
        if len(os.listdir(self.popFolder)) > 2:
            self.clear_folder()
        self.ge = GE()
        self.mutate()

        #creating a control panel
        self.info_frame = TK.Frame(root)
        self.info_frame.rowconfigure(2, weight=1)
        self.info_frame.grid(row=0)

        #(0,0) size label
        self.size_label = TK.Label(self.info_frame, text="size")
        self.size_label.grid(row=0, column=0)

        #(0,1) rule label
        self.get_current_rule(0)
        self.rule_label = TK.Label(self.info_frame, text="selected: big")
        self.mut_type = 'struct'
        self.rule_label.grid(row=0, column=1)

        #(1,0) struct label
        self.struct_label = TK.Label(self.info_frame, text="big")
        self.struct_label.grid(row=1, column=0)

        #(1,1) struct slider
        self.struct_slider = TK.Scale(self.info_frame, from_=0,
                                      to=len(self.struct_images)-1,
                                      relief=TK.RAISED,
                                      orient=TK.HORIZONTAL,
                                      command=lambda x:self.update_slider(x,'struct'))
        self.struct_slider.grid(row=1, column=1)

        #(2,0) nodal label
        self.size_label = TK.Label(self.info_frame, text="small")
        self.size_label.grid(row=2, column=0)

        #(2,1) nodal slider
        self.nodal_slider = TK.Scale(self.info_frame, from_=0,
                                     to=len(self.nodal_images)-1,
                                     orient=TK.HORIZONTAL,
                                     command=lambda x:self.update_slider(x,'nodal'))
        self.nodal_slider.grid(row=2, column=1)

        #(1,2) struct label
        txt = "of " + str(len(self.struct_images))
        self.struct_counter = TK.Label(self.info_frame, text=txt, width=5)
        self.struct_counter.grid(row=1, column=2)

        #(2,2) nodal label
        txt = "of " + str(len(self.nodal_images))
        self.nodal_counter = TK.Label(self.info_frame, text=txt, width=5)
        self.nodal_counter.grid(row=2, column=2)

        #(4,1) info label
        self.info_label = TK.Label(self.info_frame, text="Ready", width=15)
        self.info_label.grid(row=3, column=1)

        #create the individual
        name = "indiv.0"
        fullName = self.popFolder + name + '.ppm'
        tmpImg = Image.open(fullName)
        tmpImg = tmpImg.resize((self.size['imgW'], self.size['imgH']),
                               Image.ANTIALIAS)
        self.current_image = ImageTk.PhotoImage(tmpImg)
        self.indiv_frame = TK.Frame(self.root)
        self.current_indiv = TK.Label(self.indiv_frame,
                                      image=self.current_image)
        self.current_indiv.pack()
        self.indiv_frame.grid(row=0, column=1)
        self.indiv_frame.update_idletasks()

        #creating target frame
        img = Image.open(target_name)
        img = img.resize((self.size['targetW'], self.size['targetH']),
                         Image.ANTIALIAS)
        self.target_image = ImageTk.PhotoImage(img)
        self.target_frame = TK.Frame(root)
        target_indiv = TK.Label(self.target_frame, image=self.target_image)
        target_indiv.pack()
        self.target_frame.grid(row=0, column=2, sticky=TK.W)
        self.target_frame.update_idletasks()

        #bind some keys
        root.bind("<Up>", lambda event: self.change_mutation('struct'))
        root.bind("<Down>", lambda event: self.change_mutation('nodal'))
        root.bind("<Left>", lambda event: self.move_slider(-1))
        root.bind("<Right>", lambda event: self.move_slider(1))
        root.bind("<Return>", lambda event: self.regenerate())
        root.bind("<space>", lambda event: self.regenerate())
        
#######################GUI FUNCTIONS########################
    def change_mutation(self, mut_type):
        if not self.mut_type == mut_type:
            print "mutType:", mut_type
            self.mut_type = mut_type
            if mut_type == 'struct':
                self.struct_slider['relief'] = TK.RAISED
                self.nodal_slider['relief'] = TK.FLAT
                val = self.struct_slider.get()
                self.change_image(self.struct_images[val])
                self.index = 0
                self.rule_label['text'] = "selected: big"
            else:
                self.nodal_slider['relief'] = TK.RAISED
                self.struct_slider['relief'] = TK.FLAT
                val = self.nodal_slider.get()
                self.change_image(self.nodal_images[val])
                self.index = 0
                self.rule_label['text'] = "selected: small"

    def update_slider(self, val, slider):
        val = int(val)
        if slider == 'struct':
            self.change_image(self.struct_images[val])
        else:
            self.change_image(self.nodal_images[val])
        self.index = 0

    def move_slider(self, val):
        if self.mut_type == 'struct':
            newVal = self.struct_slider.get() + val
            if 0 <= newVal < len(self.struct_images):
                self.struct_slider.set(newVal)
        else:
            newVal = self.nodal_slider.get() + val
            if 0 <= newVal < len(self.nodal_images):
                self.nodal_slider.set(newVal)
        self.info_frame.update_idletasks()

    def regenerate(self):
        if self.mut_type == 'struct':
            image_no = self.struct_slider.get()
            idx = self.struct_images[image_no]['idx']
        if self.mut_type == 'nodal':
            image_no = self.nodal_slider.get()
            idx = self.nodal_images[image_no]['idx']

        print "regenerating ", idx
        #move current individual to the top of the stack
        self.ge.individuals[0] = evolver.Individual(self.ge.individuals[idx].genome)
        self.ge.individuals[0].UID = 0
        self.mutate()
        self.reset_slider()

    def reset_slider(self):
        self.struct_slider.set(0)
        self.struct_slider['to'] = len(self.struct_images)-1
        self.nodal_slider.set(0)
        self.nodal_slider['to'] = len(self.nodal_images)-1
        txt = "of " + str(len(self.struct_images)-1)
        self.struct_counter['text'] = txt
        txt = "of " + str(len(self.nodal_images)-1)
        self.nodal_counter['text'] = txt
        self.info_frame.update_idletasks()

    def change_image(self, image_data):
        idx = image_data['idx']
        rule_change = image_data['rule_change']
        distance = image_data['distance']
        print "idx:",idx, rule_change, "dist:", distance
        if (self.index + idx) >= 0 and (self.index + idx ) < self.ppm_count:
            self.index = (self.index + idx)
            fullPPMName = self.popFolder + "indiv." + str(self.index) + ".ppm"
            tmpImg = Image.open(fullPPMName)
            tmpImg = tmpImg.resize((self.size['imgW'], self.size['imgH']),
                               Image.ANTIALIAS)
            self.current_image = ImageTk.PhotoImage(tmpImg)
            self.current_indiv.configure(image=self.current_image)
            self.indiv_frame.update_idletasks()

    def mutate(self):
        #reset population and mutate
        self.clear_folder()
        for idx, indiv in enumerate(self.ge.individuals):
            if idx > 0:
                indiv.genome = None
        self.update_info("mutating", 'red')
        result = evolver.mutate_type(self.ge.individuals,
                                     self.ge.grammar)
        self.ge.individuals = result['individuals']
        self.nodal_images = result['nodal_images']
        self.struct_images = result['struct_images']

        print "nodal Count:", len(self.nodal_images)
        print "struct Count:", len(self.struct_images)
        # count the mutations and generate the images
        self.ppm_count = 0
        for indiv in self.ge.individuals:
            if not indiv.genome == None:
                self.ppm_count += 1
        self.create_ppm(self.ppm_count)

    def get_current_rule(self, codon_index):
        found = False
        for codon in self.ge.individuals[0].codon_list:
            if codon_index == codon['idx']:
                found = True
                self.current_rule = codon
                print "rule:", codon['rule'], "of type ", codon['rule_type']
                print "productions:", codon['prods']
                self.prods = codon['prods']
        if not found:
            print "NOT FOUND"

    def get_num(self, filename):
        return float(re.findall(r'\d+', filename)[0])

    def update_info(self, text, col=None):
        if col == None:
            col = self.background
        if not self.info_label == None:
            self.info_label['text'] = text
            self.info_label['bg'] = col
            self.indiv_frame.update_idletasks()

    def clear_folder(self, folder=None):
        if folder == None:
            cmd = "rm " + self.popFolder + "*.*"
        else:
            cmd = "rm " + folder
        fnull = open(os.devnull, 'w')
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE, stderr=fnull)
        process.communicate()
        fnull.close()

    def create_ppm(self, size=0):
        self.update_info("creating images", 'orange')
        if size > 0:
            cmd = (self.meditcmd + self.popFolder
                   + "indiv -a 0 " + str(size))
        else:
            cmd = (self.meditcmd + self.popFolder
                   + "indiv -a 0 " + str(self.ge.popSize))
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                                   stdin=subprocess.PIPE)
        process.communicate()
        self.update_info("Ready")
if __name__ == '__main__':
    import getopt
    TARGET = 'target1.png'
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:", ["target"])
    except getopt.GetoptError, err:
        print(str(err))
        sys.exit(2)

    for o, a in opts:
        if o in ("-t", "--target"):
            TARGET = a

    ROOT = TK.Tk()
    ROOT.title("Architype: Evolutionary Architecture")
    ROOT.geometry('+0+0')
    myGUI = GUI(ROOT, TARGET)

    ROOT.mainloop()
    print "window closed"
