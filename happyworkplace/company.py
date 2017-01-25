import worker
class Node(object):
    
    def __init__(self, me):
        self.me = me
        self.coworkers = []
        self.staff = []
        self.num_coworkers = 0
        self.num_bosses = 0
        self.num_staff = 0 
        
    def add_coworker(self, new_coworker):
        #update all coworkers
        self.coworkers.append(new_coworker)
        self.num_coworkers += 1

    def employ(self):
        print "adding employee"
        new_staff = Node(worker.Worker(self, worker.Worker.MAX_HAPPY))
        new_staff.num_bosses = self.num_bosses + 1
        print "getting coworkers"
        if self.num_staff > 0:
            new_staff.coworkers = self.staff[0].coworkers
                
        #first add this subordinate as a coworker to all staff
        print "updating coworkers"
        for staff in self.staff:
            staff.add_coworker(new_staff)

        if self.num_staff > 0:
                new_staff.num_coworkers = self.staff[0].num_coworkers

        print "adding sub"
        #now add this subordinate to the list
        self.staff.append(new_staff)
        self.num_staff += 1

    def get_boss(self):
        return self.me.boss;

    def affect_coworker(self, coworker=False):
        if not coworker: 
            coworker = self.coworkers[random.randint(0,self.num_coworkers-1)]
        self.me.affect(coworker)

    def affect_boss(self):
        self.me.affect(self.me.boss)

    def affect_subordinate(self, staff=False):
        if not staff:
            staff = self.staff[
                random.randint(0,
                               self.num_staff-1)]

        self.me.affect(staff)

    def status(self):
        print self.me.status()
        print "He has:"
        print " ", self.num_bosses, "bosses"
        print " ", self.num_staff, "staff"
        print " ", self.num_coworkers, "coworkers"
        print "==============================================="


def main():
    cls()
    boss = Node(worker.Worker())
    #boss.status()
    boss.employ()
    #boss.status()
    #boss.staff[0].status()
    boss.employ()

    boss.employ()
    boss.status()
    #boss.staff[0].status()
    for staff in boss.staff:
        staff.status()

def cls():
    #clear the screen
    print chr(27) + "[2J"

if __name__ == '__main__':
    main()
