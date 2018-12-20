import copy
import abstract
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.stats import energy_distance
from abc import ABC, abstractmethod
from functools import reduce
from itertools import chain

class Exam:
    'defines individual exams'
    def __init__(self, matnr, study, lvnumber, name, date, semester, ects, grade):
        self.matnr    = matnr
        self.study    = study
        self.lvnumber = lvnumber
        self.name     = name
        self.date     = date
        self.semester = semester
        self.ects     = float(ects)
        self.grade    = int(grade)

    def __str__(self):
        return '[' + self.name + ']'

    def __hash__(self):
        return (self.matnr, self.study, self.lvnumber, self.name, self.date, \
            self.semester, self.ects, self.grade).__hash__()


class Path:
    def __init__(self, semester, label = "", state = "none"):
        self.semester           = []
        self.courses            = []
        self.ects               = 0
        self.posEcts            = 0
        self.negEcts            = 0
        self.label              = label  #studyid
        self.state              = state

        if(semester):
          for s in semester:
              self.addSemester(s)

        self.length  = self.getLength()

    def isEmpty(self):
        return len(self.courses) == 0

    def getMeanGrade(self):
        return 0 if not len(self.courses) else reduce((lambda a, b: a+b), \
            list(map(lambda x: x.grade)))/len(self.courses)

    def getMedianGrade(self):
        if not len(self.courses): return 0
        sarr = list(map(lambda x: x.grade, self.courses))
        sarr.sort()
        mid = len(sarr)//2
        return sarr[mid]

    def getPeakSemesterECTS(self):
        semects = []
        for s in self.semester:
            if len(s):
                semects.append( reduce(lambda a,b: a+b, \
                    list(map(lambda x: x.ects, s))) )
        semects.sort()
        return semects[-1]

    def getMeanSemesterECTS(self):
        return 0 if not len(self.courses) else reduce((lambda a, b: a+b), \
            list(map(lambda x: x.ects)))/len(self.courses)

    def getFirstSemester(self):
        sem = list(map(lambda x: x.semester, self.courses))
        sem.sort()
        return sem[0]

    def getLastSemester(self):
        sem = list(map(lambda x: x.semester, self.courses))
        sem.sort()
        return sem[-1]

    def getLength(self):
        return len(self.semester)

    def getSemester(self, s):
        if len(self.semester) >= s:
            return self.semester[s-1]
        else: return None

    def setCourses(self, c):
        self.courses = c
        self.ects = reduce((lambda a,b: a+b), list(map(lambda x: x.ects, \
            self.courses))) if len(self.courses) else 0
        self.posEcts = reduce((lambda a,b: a+b), list(map(lambda x: \
            x.ects if x.grade < 5 else 0, \
            self.courses))) if len(self.courses) else 0
        self.negEcts = reduce((lambda a,b: a+b), list(map(lambda x: \
            x.ects if x.grade == 5 else 0, \
            self.courses))) if len(self.courses) else 0

    def addSemester(self, s = []):
        s = [x for x in s if x.grade < 5]
        lvnums = list(map(lambda x: x.lvnumber, self.courses))
        s = [x for x in s if x.lvnumber not in lvnums]
        #??? https://stackoverflow.com/questions/7031736/creating-unique-list-of-objects-from-multiple-lists?rq=1
        #remove double lvnumbers in semester?
        self.semester.append(s)
        self.setCourses(set().union(self.courses, s))

    def appendCoursesToSemester(self, courses, semester):
        for c in courses:
            self.appendCourseToSemester(c, semester)

    def appendCourseToSemester(self, course, semester):
        if(course.grade == 5 or
            course.lvnumber in list(map(lambda x: x.lvnumber, self.courses))):
            return
        if(len(self.semester) >= semester):
            self.semester[semester-1].append(course)
            self.setCourses(set().union(self.courses, course))

    def getPrintablePath(self):
        out = ""
        for i, s in enumerate(self.semester):
            out += "("
            for e, j in s:
                out += ", " + str(e) if j else " " + str(e)
            out += " )" if i == len(this.semester)-1 else " ) --> "
        return out

    def getPrintableShortPath(self, l = True):
        if self.label != "" and l:
            return self.label
        out = ""
        for i, s in enumerate(self.semester):
            for e in s:
                out += str(e)
            out += "" if len(self.semester)-1 == i else "-\n"
        return out

    def __str__(self):
        return self.getPrintableShortPath(False)


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
class RaphMetric(AbstractMetric):
    def __init__(self, paths):
        self.distanceMatrix = None
        self.distanceDict   = dict()
        self.paths          = sorted(paths, key=lambda x: x.label)
        self.pathNames      = sorted(list(map(lambda x: x.label, paths)))
        self.allCourseNames = extractAllCourseNames(paths)
        np.set_printoptions(suppress=True)

    def calculateDistanceDict(self):
        #create array of all courses, sorted
        #create vector from lookuptable (sorted courses)
        #vector - each value as new internal representation matrix
        #transpose before output
        self.distanceDict = dict()
        for path in self.paths:
            distRow = dict()
            for path2 in self.paths:
                if path.label == path2.label:
                    distRow[path2.label] = 0
                else:
                    distRow[path2.label] = self.calculatePathDistance(path, path2)
            self.distanceDict[path.label] = distRow
        return self.distanceDict

    def calculateDistance(self):
        self.distanceMatrix = None
        for i, path in enumerate(self.paths):
            distRow = np.array([])
            for j, path2 in enumerate(self.paths):
                if j <= i:
                    distRow = np.append(distRow, [0])
                else:
                    distRow = np.append(distRow, [self.calculatePathDistance(path, path2)])
            if self.distanceMatrix is None:
                self.distanceMatrix = np.array([distRow])
            else:
                self.distanceMatrix = np.vstack((self.distanceMatrix, distRow))
        i_lower = np.tril_indices(len(self.pathNames), -1)
        self.distanceMatrix[i_lower] = self.distanceMatrix.T[i_lower]
        return self.distanceMatrix, self.pathNames

    def calculatePathDistance(self, pathA, pathB):
        courseNames = extractAllCourseNames([pathA, pathB])
        semesterA, semesterB = createExtendedLookUpList(pathA, pathB, courseNames)
        distanceA = self.generateDistanceMatrix(semesterA)
        distanceB = self.generateDistanceMatrix(semesterB)
        distanceDiff = np.sign(np.subtract(distanceA, distanceB))
        distance = np.sum(np.absolute(distanceDiff))/2
        return distance

    def generateDistanceMatrix(self, semesterVector):
        out = []
        for el in semesterVector:
            out.append(np.subtract(semesterVector, el))
        return np.transpose(np.sign(out))

class RaphMetricImproved(abstract.AbstractMetric):
    def __init__(self, paths):
        self.distanceMatrix = None
        self.distanceDict   = dict()
        self.paths          = sorted(paths, key=lambda x: x.label)
        self.pathNames      = sorted(list(map(lambda x: x.label, paths)))
        self.allCourseNames = abstract.extractAllCourseNames(paths)
        np.set_printoptions(suppress=True)

    def calculateDistanceDict(self):
        #create array of all courses, sorted
        #create vector from lookuptable (sorted courses)
        #vector - each value as new internal representation matrix
        #transpose before output
        self.distanceDict = dict()
        for path in self.paths:
            distRow = dict()
            for path2 in self.paths:
                if path.label == path2.label:
                    distRow[path2.label] = 0
                else:
                    distRow[path2.label] = self.calculatePathDistance(path, path2)
            self.distanceDict[path.label] = distRow
        return self.distanceDict

    def calculateDistance(self):
        self.distanceMatrix = None
        for i, path in enumerate(self.paths):
            distRow = np.array([])
            for j, path2 in enumerate(self.paths):
                if j <= i:
                    distRow = np.append(distRow, [0])
                else:
                    distRow = np.append(distRow, [self.calculatePathDistance(path, path2)])
            if self.distanceMatrix is None:
                self.distanceMatrix = np.array([distRow])
            else:
                self.distanceMatrix = np.vstack((self.distanceMatrix, distRow))
        #i_lower = np.tril_indices(len(self.pathNames), -1)
        #self.distanceMatrix[i_lower] = self.distanceMatrix.T[i_lower]
        return self.distanceMatrix, self.pathNames

    def calculatePathDistance(self, pathA, pathB):
        coursesA = abstract.extractAllCourseNames([pathA])
        coursesB = abstract.extractAllCourseNames([pathB])
        courseNames = abstract.extractAllCourseNames([pathA, pathB])
        semesterA, semesterB = abstract.createExtendedLookUpList(pathA, pathB, courseNames)
        distanceA = self.generateDistanceMatrix(semesterA)
        distanceB = self.generateDistanceMatrix(semesterB)
        distanceA = self.setNonOverlap(distanceA, courseNames, coursesA, 0)
        distanceB = self.setNonOverlap(distanceB, courseNames, coursesB, 1)
        distanceDiff = np.absolute(np.sign(np.subtract(distanceA, distanceB)))


        n, m = distanceDiff.shape
        size = n - 1
        tri = np.fliplr(distanceDiff)[np.triu_indices(size)]
        distance = sum(tri)/len(tri)
        return distance

    def setNonOverlap(self, distanceMatrix, courseNames, ownCourses, val):
        for i, course in enumerate(courseNames):
            if course not in ownCourses:
                self.replaceRowCol(distanceMatrix, i, val)
        return distanceMatrix

    def replaceRowCol(self, M, i, val):
        nrows, ncols = M.shape
        #if(index>nrows or index>ncols) return
        x = np.full([1, ncols], val)
        y = np.full([nrows, 1], val)
        M[i,:] = x
        M[:,i:(i+1)] = y
        np.fill_diagonal(M, 0)
        return M

    def generateDistanceMatrix(self, semesterVector):
        out = []
        for el in semesterVector:
            out.append(np.subtract(semesterVector, el))
        return np.transpose(np.sign(out))

class EarthMoversMetric(AbstractMetric):
    def __init__(self, paths):
        self.distanceMatrix = None
        self.distanceDict   = dict()
        self.paths          = sorted(paths, key=lambda x: x.label)
        self.pathNames      = sorted(list(map(lambda x: x.label, paths)))
        self.allCourseNames = extractAllCourseNames(paths)
        np.set_printoptions(suppress=True)

    def calculatePathDistance(self, pathA, pathB):
        courseNames = extractAllCourseNames([pathA, pathB])
        semesterA, semesterB = createExtendedLookUpList(pathA, pathB, courseNames)
        distance = wasserstein_distance(semesterA, semesterB)
        return distance

    def calculateDistance(self):
        self.distanceMatrix = None
        for i, path in enumerate(self.paths):
            distRow = np.array([])
            for j, path2 in enumerate(self.paths):
                if j <= i:
                    distRow = np.append(distRow, [0])
                else:
                    distRow = np.append(distRow, [self.calculatePathDistance(path, path2)])
            if self.distanceMatrix is None:
                self.distanceMatrix = np.array([distRow])
            else:
                self.distanceMatrix = np.vstack((self.distanceMatrix, distRow))
        i_lower = np.tril_indices(len(self.pathNames), -1)
        self.distanceMatrix[i_lower] = self.distanceMatrix.T[i_lower]
        return self.distanceMatrix, self.pathNames

    def calculateDistanceDict(self):
        self.distanceDict = dict()
        for path in self.paths:
            distRow = dict()
            for path2 in self.paths:
                if path.label == path2.label:
                    distRow[path2.label] = 0
                else:
                    distRow[path2.label] = self.calculatePathDistance(path, path2)
            self.distanceDict[path.label] = distRow
        return self.distanceDict


class EnergyDistanceMetric(AbstractMetric):
    def __init__(self, paths):
        self.distanceMatrix = None
        self.distanceDict   = dict()
        self.paths          = sorted(paths, key=lambda x: x.label)
        self.pathNames      = sorted(list(map(lambda x: x.label, paths)))
        self.allCourseNames = extractAllCourseNames(paths)
        np.set_printoptions(suppress=True)

    def calculateDistanceDict(self):
        self.distanceDict = dict()
        for path in self.paths:
            distRow = dict()
            for path2 in self.paths:
                if path.label == path2.label:
                    distRow[path2.label] = 0
                else:
                    distRow[path2.label] = self.calculatePathDistance(path, path2)
            self.distanceDict[path.label] = distRow
        return self.distanceDict

    def calculateDistance(self):
        self.distanceMatrix = None
        for i, path in enumerate(self.paths):
            distRow = np.array([])
            for j, path2 in enumerate(self.paths):
                if j <= i:
                    distRow = np.append(distRow, [0])
                else:
                    distRow = np.append(distRow, [self.calculatePathDistance(path, path2)])
            if self.distanceMatrix is None:
                self.distanceMatrix = np.array([distRow])
            else:
                self.distanceMatrix = np.vstack((self.distanceMatrix, distRow))
        i_lower = np.tril_indices(len(self.pathNames), -1)
        self.distanceMatrix[i_lower] = self.distanceMatrix.T[i_lower]
        return self.distanceMatrix, self.pathNames

    def calculatePathDistance(self, pathA, pathB):
        courseNames = extractAllCourseNames([pathA, pathB])
        semesterA, semesterB = createExtendedLookUpList(pathA, pathB, courseNames)
        distance = energy_distance(semesterA, semesterB)
        return distance

def raphRaphDistFun(pathA, pathB):
    metric = RaphMetric([pathA, pathB])
    return metric.calculatePathDistance(pathA, pathB)

def raphRaphImDistFun(pathA, pathB):
    metric = RaphMetricImproved([pathA, pathB])
    return metric.calculatePathDistance(pathA, pathB)

def raphEarthDistFun(pathA, pathB):
    metric = EarthMoversMetric([pathA, pathB])
    return metric.calculatePathDistance(pathA, pathB)

def raphEnergyDistFun(pathA, pathB):
    metric = EnergyDistanceMetric([pathA, pathB])
    return metric.calculatePathDistance(pathA, pathB)
