import copy
from rlcard.utils.utils import init_standard_deck
from operator import add
import rlcard.games.limitholdem.utils as utils
from random import sample
from rlcard.games.base import Card
import json
HOWMANY = {'p':0,'f':3,'t':4, 'r':5}
class Analyst:
    #key is to_String of set of cards
    winprob = {}
    categorydist = {}
    oppcategorydist = {}
    def __init__(self, N = 10000, games = None):
        self.winprob = {}
        self.categorydist = {}
        self.oppcategorydist = {}
        self.N = N
        self.gamefile = games
        #initialize the dictionaries
    def cards_to_String (self, all_cards):
        '''
        Sorts the cards and then concatonates them with commas as delimiters
        '''
        all_cards = sorted(all_cards, key = lambda card: card.__hash__())
        return ','.join([card.get_index() for card in all_cards])
    def to_String(self, hole_cards,public_cards):
        return self.cards_to_String(hole_cards)+";"+self.cards_to_String(public_cards)
    def comphands(self, p1cards, p2cards):
        dist = utils.compare_hands([list(map(Card.get_index,p1cards)),list(map(Card.get_index,p2cards))])
        return (1+dist[0]-dist[1])/2
    def win_prob(self, hole_cards, public_cards, n):
        '''
        HEADS UP ONLY:
        Computes the win probability recursively, storing in a dictionary
        Args:
            hole_cards: Private cards of the protagonist
            public_cards: The public cards so far
            n: number of public cards (between 0 and 5)
        Returns:
            probability of winning
            - just do this over N trials of picking random remaining cards
        '''
        #if we already computed (DICT ONLY STORES flop and preflop)
        if self.to_String(hole_cards, public_cards) in self.winprob:
            return self.winprob[self.to_String(hole_cards, public_cards)]

        remaining_cards = init_standard_deck()
        all_cards = copy.deepcopy(hole_cards + public_cards)
        for card in all_cards:
            remaining_cards.remove(card)
        p = 0
        c = 0
        ntrials = self.N
        if n == 0:
            ntrials = max (ntrials, 1000)
        for _ in range(ntrials):
            drawrandom = sample(remaining_cards,7-n)
            p += self.comphands(all_cards+drawrandom[2:], public_cards + drawrandom)
        p/= ntrials
        #store and return!
        if n == 0 or n == 3:
            self.winprob[self.to_String(hole_cards, public_cards)] = p
        return p
    def category_distribution(self, hole_cards, public_cards, opp = False):
        '''
        HEADS UP ONLY:
        Computes probability distribution of categories (high card, pair, two pair, three of a kind, etc
        Args:
            hole cards
            public cards
            opp (for opponent or for us?)
            n: number of total cards (between 0 and 7)
        Returns:
            prob distribution
        '''
        #the general framework is: all_cards is what we have so far, remaining_cards is the set of cards that we could get
        #if for the protagonist, we know the hole cards so just do that
        all_cards = copy.deepcopy(public_cards)
        if not opp:
            all_cards += hole_cards
        #if we already computed this (DICT STORES ONLY preflop and flop)
        if opp:
            if self.to_String(hole_cards,public_cards) in self.oppcategorydist:
                return self.oppcategorydist[self.to_String(hole_cards,public_cards)]
        else:
            if self.to_String(hole_cards,public_cards) in self.categorydist:
                return self.categorydist[self.to_String(hole_cards,public_cards)]

        remaining_cards = init_standard_deck()
        for card in all_cards:
            remaining_cards.remove(card)
        if opp:
            for card in hole_cards:
                remaining_cards.remove(card)
        #what we end up returning
        dist = [0 for _ in range(9)]
        
        n = len(all_cards)
        for _ in range(self.N):
            drawrandom = sample(remaining_cards, 7-n)
            hand = utils.Hand(list(map(Card.get_index,drawrandom+all_cards)))
            hand.evaluateHand()
            dist[hand.category-1]+=1
        dist = [x/self.N for x in dist]

        if opp and (n == 0 or n== 3):
            self.oppcategorydist[self.to_String(hole_cards,public_cards)] = dist
        if not opp and (n == 2 or n == 5): 
            self.categorydist[self.to_String(hole_cards,public_cards)] = dist
        return dist
    def card_parameters(self, hole_cards, public_cards):
        '''
        Computes the parameters to be fed into the model
        '''
        winp = self.win_prob(hole_cards, public_cards,len(public_cards))
        owndist = self.category_distribution(hole_cards, public_cards)
        oppdist = self.category_distribution(hole_cards, public_cards, opp = True)
        return [winp]+owndist+oppdist
    #INPROGRESS
    def stage_parameters(self, pcards, truncboard, bets):
        return 0
    #given an elem, return board, playercards, roundaggression, betsperstage
    def parameters(self, elem):
        #elem is an element of the file handsup.json
        #extract hole_cards from whoever's turn it is, public_cards, whether they are small blind or big blind
        #aggression score each round (number of bets you did so far)

        #store the whole board and the hole cards for each player (done :D)
        board = [Card(s[1].upper(),s[0]) for s in elem['board']]
        playercards = [[],[]]
        roundaggression = [{},{}]
        betsperstage = [{},{}]
        goodround = True
        lastround = 's'
        for player in elem['players']:
            playercards[elem['players'][player]['position']-1] = [Card(s[1].upper(),s[0]) for s in elem['players'][player]['pocket_cards']]
            for bet in elem['players'][player]['bets']:
                betsperstage[elem['players'][player]['position']-1][bet['stage']] = bet['actions']
                roundaggression[elem['players'][player]['position']-1][bet['stage']] = bet['actions'].count('b')+bet['actions'].count('r')
                if 'K' in bet['actions'] or 'Q' in bet['actions']:
                    goodround = False
                if 'f' in bet['actions']:
                    roundaggression[elem['players'][player]['position']-1][bet['stage']] -= 1
                    if lastround == 's' or HOWMANY[bet['stage']] < HOWMANY[lastround]:
                        lastround = bet['stage']
                if 'A' in bet['actions']:
                    lastround = bet['stage']
        #board is the board we saw (not even necessarily the whole board)
        #playercards are the hole-cards of each player (empty if unseen)
        #betsperstage is a list of two dicts. Each dict maps the stage to the string of actions that occurred that stage for that player
        #each element of roundagression is a dict mapping the stage to paggro that round
        #each element of betsperstage is a dict mapping stage to bets for each player
        #lastround is the last round before someone folded (this is one of 'p','f', 't', 'r', 's', 'X') where 's' means we got to showdown, 'X' means round is invalid
        #to compute this, we look at betsperstage for the first stage with a fold action
        return {'board':board, 'pcards': playercards, 'aggro':roundaggression, 'bets':betsperstage, 'lastround': lastround if goodround else 'X'}
    #gda_parameters:
    #fin: a json file with only datapoints that went to showdown
    #fout: name of json file we are making
    #given fin, this method writes to fout a list of (stage, aggscore, winprob 2-vector)
    #aggscore per round is total number of bets, so a list of two integers
    def gda_parameters(self, fin, fout):
        #for each element of fin, compute its parameters.
        with (open(fin)) as g:
            data = json.load(g)
        paramlist = []
        c = 0
        for elem in data:
            c+=1
            if (c%10 == 0): print(c)
            info = self.parameters(elem)
            aggsofar = [0, 0]
            #for each stage of the game, from preflop to river,
            for stage in ['p', 'f', 't', 'r']:
                #for each player,
                winprob = [0,0]
                for i in range(2):
                    #compute aggression score (sum of roundaggro so far) and winprobs (winprob in truncated board)
                    aggsofar[i] += info['aggro'][i][stage]
                    winprob[i] = self.win_prob(info['pcards'][i], info['board'][:HOWMANY[stage]], HOWMANY[stage])
                paramlist.append([stage,aggsofar[0],aggsofar[1],winprob[0],winprob[1]])
        print(c)
        with (open(fout,'w')) as f:
            json.dump(paramlist, f)
    #filters the elems in games, storing only the ones that went to showdown in fout
    def filter(self, fout):
        with open(self.gamefile) as f:
            data = json.load(f)
        filtered = []
        c = 0
        for elem in data:
            c+=1
            info = self.parameters(elem)
            #if last round is showdown
            if info['lastround'] == 's':
                filtered.append(elem)
        with open(fout,'w') as g:
            json.dump(filtered,g)
if __name__ == '__main__':
    s = 'C:/Users/eax20/Downloads/rlcard-master/rlcard-master/rlcard/games/limitholdem/'
    with open(s+'headsup1.json') as f:
        data = json.load(f)
    trunc = []
    for _ in range(10000):
        trunc.append(data[_])
    with open(s+'headsup1.json','w') as g:
            json.dump(trunc, g)
    a = Analyst(400, s+'headsup1.json')
    a.filter(s+'headsup1showdown.json')
    print("filter done")
    a.gda_parameters(s+'headsup1showdown.json',s+'showdowndata1.json')