import numpy as np
import math
import matplotlib.pyplot as plt
import json
import copy
from rlcard.utils.utils import init_standard_deck
from rlcard.games.base import Card
import random
import analyst as an
HOWMANY = {'p':0,'f':3,'t':4, 'r':5}
STAGES = ['p','f','t','r']
def logit(p):
    return np.log(p) - np.log(1 - p)
class HoleCardModel:
    #mu is a and sigma are dictionaries mapping stages and aggression scores to the parameters of the corresponding gaussian distribution
    def __init__ (self, fin, fout, M = 10, N = 30, epsilon = 0.1):
        self.fin = fin
        self.fout = fout
        self.mu = {}
        self.sigma = {}
        self.datapoints = {}
        self.epsilon = epsilon
        self.analyst = an.Analyst(N = 400)
        self.N = N
        self.M = M
    def plot (self, stage, aggscore = None):
        stagecolor = {'p' :'g', 'f' : 'b', 't' : 'r', 'r' : 'k'}
        with open(self.fin, 'r') as f:
            data = json.load(f)
        if aggscore:
            pdata = [i for i in data if i[0] == stage and aggscore == [i[1],i[2]]]
        else:
            pdata = [i for i in data if i[0] == stage]
        plt.scatter([logit(i[3]) for i in pdata], [logit(i[4]) for i in pdata], c = [(stagecolor[stage], min(1,0.5+(i[1]+i[2])/20)) for i in pdata])
        stagelabel = {'p':'Preflop', 'f': 'Flop', 't': 'Turn', 'r':'River'}
        agglabel = ''
        if aggscore:
            plt.title(stagelabel[stage]+' Quality Distribution for Aggression '+str(aggscore[0])+', '+str(aggscore[1]))
        else:
            plt.title(stagelabel[stage]+' Quality Distribution')
        plt.xlabel("Logit of Small Blind Hand Quality")
        plt.ylabel("Logit of Big Blind Hand Quality")
        plt.show()
        return
    def run (self):
        with open(self.fin, 'r') as f:
            data = json.load(f)
        for stage in ['p','f','t','r']:
            self.datapoints[stage] = [None]*5
            for i in range(5):
                self.datapoints[stage][i] = [None]*5
                for j in range(5):
                    self.datapoints[stage][i][j] = [[_[3],_[4]] for _ in data if _[0] == stage and min(4, _[1]) == i and min(4, _[2]) == j]
                    print(stage, i, j, len(self.datapoints[stage][i][j]))
        return 0
    #given a game aggscore, predict a set of holecards for the two players that are within epsilon of acceptable parameters
    #how to do this?? call a predict_helper submethod that does it for a single player given a winprob and bet history
    #how to do that?? choose a random datapoint and look at its winprob values AT EACH STAGE. Then look at possible hole cards worst case, try all 52C2 hole cards
    def random_sample(self, stage, aggscore,  passive = [False, False], n = 1):
        x = min(4,aggscore[0])
        y = min(4,aggscore[1])
        x = max (x, 0)
        y = max (y, 0)
        if (passive[0]):
            while (n > len(self.datapoints[stage][x][y])):
                    x = random.randint(0, 4)
                    y = random.randint(0, 4)
            #take random sample from [0, x] basically
            samplex = random.sample([v[0] for v in self.datapoints[stage][x][y]], n)
            sampley = random.sample([v[1] for l in self.datapoints[stage][x][y:] for v in l], n)
            return [[samplex[_], sampley[_]] for _ in range(n)]
        if (passive[1]):
            while (n > len(self.datapoints[stage][x][y])):
                    x = random.randint(0, 4)
                    y = random.randint(0, 4)
            samplex = random.sample([v[0] for z in range(x, 5) for v in self.datapoints[stage][z][y]], n)
            sampley = random.sample([v[1] for v in self.datapoints[stage][x][y]], n)
            return [[samplex[_], sampley[_]] for _ in range(n)]
        if (n > len(self.datapoints[stage][x][y])):
            if x > y:
                return self.random_sample(stage, [x, y], [False, True], n)
            elif x < y:
                return self.random_sample(stage, [x, y], [True, False], n)
            else:
                while (n > len(self.datapoints[stage][x][y])):
                    x = random.randint(0, 4)
                    y = random.randint(0, 4)
        return random.sample(self.datapoints[stage][x][y], n)
    # distance as a vector
    def d(self, v1, v2):
        return np.linalg.norm(np.array(v1)-np.array(v2))
    #
    def twodist(self, stage, desiredwps, truewps, passive):
        mindst = {}
        squaresum = 0
        for s in STAGES[:STAGES.index(stage)+1]:
            if stage == s:
                #check passivity: for the passive player, instead of min distance compute
                folder = 0 if passive[0] else 1
                aggressor = 1 - folder
                x = max([truewps[s][folder] - dwp[folder] for dwp in desiredwps[s]])
                x = max(0, x)
                y = min ([abs(truewps[s][aggressor] - dwp[aggressor]) for dwp in desiredwps[s]])
                mindst[s] = math.sqrt(x**2+y**2)
            else:
                mindst[s] = min([self.d(truewps[s],dwp) for dwp in desiredwps[s]])
            squaresum+= mindst[s]**2
        return math.sqrt(squaresum)
    # minimum distance from set of desiredwps and our choice of cards
    def twodistance(self, stage, desiredwps, board, p1cards, p2cards, passive):
        truewps = {}
        for i in range(STAGES.index(stage)+1):
            n = HOWMANY[STAGES[i]]
            truewps[STAGES[i]] = [self.analyst.win_prob(p1cards, board[:n], n), self.analyst.win_prob(p2cards, board[:n], n)]
        return self.twodist(stage, desiredwps, truewps, passive)
    #distance from a single point in win problem
    def dist (self, stage, desiredwp, board, cards):
        distvect = [0]*(STAGES.index(stage)+1)
        for i in range(len(distvect)):
            n = HOWMANY[STAGES[i]]
            distvect[i] = self.analyst.win_prob(cards, board[:n], n) - desiredwp[STAGES[i]]
        return np.linalg.norm(np.array(distvect))
    def predict (self, elem):
        info = self.analyst.parameters(elem)
        #find desiredwinprobs, the multiple desired winprobabilities at each stage of the game
        #do nothing or return null if either we got to showdown or this is an invalid element
        laststage = info['lastround']
        if laststage not in STAGES:
            if laststage == 's':
                return 1
            else:
                return None
        #basically i need a "randomlessthan(desiredwps)"
        desiredwps = {}
        passive = [False, False]
        aggscore = [0,0]
        for stage in STAGES[:STAGES.index(laststage)+1]:
            #compute the aggscore so we know from where to sample
            for i in range(2):
                if(info['aggro'][i][stage] == -1):
                    passive[i] = True
                else:
                    aggscore[i] += info['aggro'][i][stage]
            desiredwps[stage] = self.random_sample(stage, aggscore, passive if stage == laststage else [False, False], self.M)
        #remove from the cards in the board from the set of all cards
        remaining_cards = init_standard_deck()
        for card in info['board']:
            remaining_cards.remove(card)
        '''
        dist = [0]*2
        bestcards = [0]*2
        result = self.closest_card(laststage, desiredwp[0], remaining_cards, info['board'])
        dist[0] = result[0]
        bestcards[0] = result[1]
        for card in result[1]:
            remaining_cards.remove(card)
        result = self.closest_card(laststage, desiredwp[1], remaining_cards, info['board'])
        dist[1] = result[0]
        bestcards[1] = result[1]
        return bestcards
        '''
        #compute desiredwps
        bestdist = math.inf
        bestcards = random.sample(remaining_cards, 4)
        for _ in range(self.N):
            cards = random.sample(remaining_cards, 4)
            #if passive, instead of distance the metric is: distance for non-passive player, and max (0, winprob - dwp)
            dist = self.twodistance(laststage, desiredwps, info['board'], cards[:2], cards[2:], passive)
            if (dist < bestdist):
                bestdist = dist
                bestcards = copy.deepcopy(cards)
            if (bestdist < self.epsilon):
                break
        return [bestcards[:2], bestcards[2:]]
    def closest_card(self, laststage, desiredwp, remaining_cards, board):
        mindist = math.inf
        bestcards = []
        for _ in range(self.N):
            print(_)
            print(mindist)
            cards = random.sample(remaining_cards, 2)
            #dist computes difference between desiredwp and actual winp of cards
            dist = self.dist(laststage, desiredwp, board, cards)
            if (dist < mindist):
                mindist = dist
                bestcards = copy.deepcopy(cards)
            if (mindist < self.epsilon):
                break
        return [mindist, bestcards]
        
    def guess(self, rawfile, fileout):
        with open(rawfile, 'r') as f:
            data = json.load(f)
        c = 0
        newdata = []
        for elem in data:
            result = self.predict(elem)
            c += 1
            if result:
                if result == 1:
                    newdata.append(elem)
                else:
                    s = [list(map(lambda x: x.rank + (x.suit).lower(), cards)) for cards in result]
                    for p in elem["players"]:
                        elem["players"][p]["pocket_cards"] = s[elem["players"][p]["position"]-1]
                    print(c)
                    newdata.append(elem)
        with open(fileout, 'w') as g:
            json.dump(newdata, g)
                
if __name__ == '__main__':
    s = 'C:/Users/eax20/Downloads/rlcard-master/rlcard-master/rlcard/games/limitholdem/'
    hcm = HoleCardModel(s+'showdowndata1.json', s+'hcmparams1.json')
    hcm.plot('p', [0,0])
    hcm.plot('p')
    hcm.plot('f')
    hcm.plot('t')
    hcm.plot('r')
    hcm.run()
    print("run done!")
    hcm.guess(s+'headsup1.json', s+'headsup1completed.json')