import time, random, namegen

class Worker():

    MIN_HAPPY = 0
    MAX_HAPPY = 100
    PROMOTION_GAIN = 10
    NAME_GEN = namegen.NameGen('./name_gen/Languages/elven.txt')
    MIN_EFFECT_UP = 0
    MAX_EFFECT_UP = 0.5
    MIN_EFFECT_ACROSS = 0.25
    MAX_EFFECT_ACROSS = 0.75
    MIN_EFFECT_DOWN = 0.5
    MAX_EFFECT_DOWN = 1
    # @TODO
    # Add a resistance to effect value
    
    def __init__(self, boss=None, happiness=100, name=False):
        if( not name ): name = self.NAME_GEN.gen_word()
        self.name = name
        self.happiness = happiness
        self.boss = boss
        self.effect_up = round(random.uniform(self.MIN_EFFECT_UP, 
                              self.MAX_EFFECT_UP),2)
        self.effect_down = round(random.uniform(self.MIN_EFFECT_DOWN, 
                                self.MAX_EFFECT_DOWN),2)
        self.effect_across = round(random.uniform(self.MIN_EFFECT_ACROSS, 
                                  self.MAX_EFFECT_ACROSS),2)

    def update(self, happiness):
        self.happiness = happiness
        self.correct_happiness()
        self.happiness = round(self.happiness,1)

    def promote(self):
        self.level += 1
        self.happiness += happiness / Worker.PROMOTION_GAIN

    def disposition(self, amount):
        self.happiness += amount
        self.correct_happiness()

    def affect(self, worker):
        outlook = self.happiness - worker.happiness
        effect = outlook * self.effect_across
        worker.update(worker.happiness + effect)         

    def correct_happiness(self):
        if self.happiness > Worker.MAX_HAPPY: self.happiness = Worker.MAX_HAPPY
        if self.happiness < Worker.MIN_HAPPY: self.happiness = Worker.MIN_HAPPY

    def status(self):
        status = self.name + " is " + str(self.happiness) + '% happy ( up=' + str(self.effect_up) + '; down=' + str(self.effect_down) + '; across=' + str(self.effect_across) + ')'
        return status

def main():
    workers = []
    for i in range(10):
        workers.append(Worker(None,
                              random.random(),
                              random.randint(0,100)))
        
    num_workers = len(workers)
    i=0
    while True:
        happy_total = 0
        for worker in workers:
            worker.affect(workers[random.randint(0, num_workers-1)])
            worker.status()
            time.sleep(0.1)
            happy_total += worker.happiness
        
        avg_happiness = happy_total / num_workers
        print "AVERAGE HAPPINESS =", str(avg_happiness) + "%"
        cls()
        i += 1
                
        #myWorker.status()
        #myWorker.disposition(-20)
    # while True:
    #     myWorker.update(random.randint(0,100))
    #     myWorker.status()
    #     time.sleep(1)
    
    #myWorker.status()

def cls():
    #clear the screen
    print chr(27) + "[2J"
    
if __name__ == '__main__':
    main()



    
