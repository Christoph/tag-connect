import copy
from functools import reduce
from itertools import chain

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
        return self.getPrintableShortPath()
