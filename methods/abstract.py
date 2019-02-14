from abc import ABC, abstractmethod

def extractAllCourseNames(paths):
    out = set([])
    for p in paths:
        courseNames = set(map(lambda x: x.name, p.courses))
        out = set().union(courseNames, out)
    return sorted(out)

def createExtendedLookUpList(pathA, pathB, courseNames):
    lookUpA, lookUpB = [], []
    for n in courseNames:
        for indexA, semA in enumerate(pathA.semester):
            namesA = list(map(lambda x: x.name, semA))
            if n in namesA:
                lookUpA.append(indexA)
                break
            if indexA == len(pathA.semester)-1:
                lookUpA.append(len(pathA.semester))
        for indexB, semB in enumerate(pathB.semester):
            namesB = list(map(lambda x: x.name, semB))
            if n in namesB:
                lookUpB.append(indexB)
                break
            if indexB == len(pathB.semester)-1:
                lookUpB.append(len(pathB.semester))
    return lookUpA, lookUpB


class AbstractMetric(ABC):
    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def calculateDistance(self):
        pass
