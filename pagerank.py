import os
import random
import re
import sys
import numpy as np

# Sampling
# Random Number Generator generates number between 0 and 1, if above .85, choose random website
# Else, choose from links on page
# Keep track of how many times each page has been visited, divide each by SAMPLES and return

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print("PageRank Results from Sampling (n = {SAMPLES})".format(SAMPLES=SAMPLES))
    for page in sorted(ranks):
        print("  {page}: {ranks}".format(page=page, ranks = ranks[page]))
    ranks = iterate_pagerank(corpus, DAMPING)
    print("PageRank Results from Iteration")
    for page in sorted(ranks):
        print("  {page}: {ranks}".format(page=page, ranks = round(ranks[page],5)))


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    pages = [i for i in corpus.keys()]
    pageProbs = {}
    #initializes the page probability dictionary 
    for p in pages:
        pageProbs[p] = (1-damping_factor)*1/len(pages) 
    
    connectsTo = corpus[page] #pages website links to
    for connected in connectsTo:
        pageProbs[connected] += damping_factor*1/len(connectsTo)
    
    return pageProbs    


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = [i for i in corpus.keys()]
    pageRanks = {}
    for p in pages:
        pageRanks[p] = 0
    randomPage = random.choice(pages)
    for x in range(n):
        #print(randomPage)
        #print("##########")
        model = transition_model(corpus, randomPage, damping_factor)
        #print(randomPage, model)
        keys, values = [i for i in model.keys()], [i for i in model.values()]
        #print(keys,values)
        #chosen = np.random.choice(keys, 1, values)[0]
        chosen = ''
        myRandomNumber = random.random()
        cumulativeValues = np.cumsum(values)
        #print(cumulativeValues)
        for i,x in enumerate(cumulativeValues):
            if(myRandomNumber <= x):
                chosen = keys[i]
                break
        if(chosen == ''):
            print(myRandomNumber,keys,values)
            print("Something is very wrong")
            raise NotImplementedError
        
        pageRanks[chosen]+=1
        #print(chosen)
        #print("##########")
        randomPage = chosen
        
    for z in pages:
        pageRanks[z] = pageRanks[z]/n

    return pageRanks        



def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages, links = [i for i in corpus.keys()],[i for i in corpus.values()]
    #print(pages,links)
    #initialize pageRanks to 1/N
    pageDicts = {}
    for pg in pages:
        pageDicts[pg] = 1/len(pages)

    def getPageRank(page,dicts,corp):
        allThingsThatLinkToThisPage = []
        for i in corp:
            if(page in corp[i]):
                allThingsThatLinkToThisPage.append(i)
        
        #print(allThingsThatLinkToThisPage)
        toAdd = [dicts[linkPage]/len(corp[linkPage]) for linkPage in allThingsThatLinkToThisPage]
        #print(len([i for i in corp.keys()]))
        return (1-damping_factor)/len([i for i in corp.keys()]) + damping_factor*sum(toAdd)
    
    for i in range(100000):
        for x in pages:
            #print(x)
            pageDicts[x] = getPageRank(x, pageDicts,corpus)
        #break
    return pageDicts
    #raise NotImplementedError

if __name__ == "__main__":
    main()
