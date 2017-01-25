import random

flip_count = 1000

def flip_coin(amount):
    total_flips = amount
    score = 0
    previous = 0
    heads = 0 
    tails = 0
    losing = 0
    winning = 0
    TF_site = 0
    P_site = 0
    TF_found = 0
    P_found = 0
    longest_winning = 0
    longest_losing = 0

    result = open('results.txt','w')
    for i in range(amount):
        coinflip = random.randint(0,1)
        if coinflip == 1:
            heads += 1
            if coinflip == previous:
                winning += 1
                TF_site += 1

                if TF_site > 7: 
                    TF_found += 1
                    total_flips += 224
                    TF_site = 0

                if winning > longest_winning:
                    longest_winning = winning
            else:
                winning = 1
                TF_site = 1
                previous = coinflip	
            score += 1
        else:
            tails += 1
            if coinflip == previous:
                losing += 1
                P_site += 1

                if P_site > 7: 
                    total_flips += 224
                    P_found += 1
                    P_site = 0

                if losing > longest_losing:
                    longest_losing = losing
            else:
                losing = 1
                P_site = 1
                previous = coinflip
            score -= 1
        #print "score ",score
        result.write(str(i) + " " + str(score)  + '\n')
    print "heads total:", heads, "percent:", str(int((100.0/amount)*heads))
    print "tails total:", tails, "percent:", str(int((100.0/amount)*tails))
    print "win streak: ", longest_winning
    print "lose streak:", longest_losing
    print "TF found:", TF_found, "P found", P_found
    used_codons = (TF_found + P_found) * 256
    percent_used = (100.0 / total_flips) * used_codons
    print "used:", used_codons, "out of",total_flips,"percentage:", percent_used
    result.close()

flip_coin(flip_count)
