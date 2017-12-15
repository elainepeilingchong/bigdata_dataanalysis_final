from pprint import pprint
import csv
import json
from bson import json_util
from pymongo import MongoClient
import redis
from py2neo import Graph, Node, Relationship, authenticate,Path

#READ FILE
f = ("studentid","lab_1","christmas_test","lab_2","easter_test", "lab_3","part_time_job","exam_grade")
json_path = '/Users/elainechong/Desktop/Sem_5/Final_CA/Elaine_Chong/FinalProjectData.json'
csvfile = open('/Users/elainechong/Desktop/Sem_5/Final_CA/Elaine_Chong/FinalProjectData.csv', newline='')
reader = csv.DictReader(csvfile, delimiter=';', fieldnames=f)
json_list=[]
reader.__next__()
for row in reader:
    json_list.append(row)

#write json into file
json_data = json.dumps(json_list,sort_keys=True, indent=4)

fout = open(json_path,"w")
fout.write(json_data)
fout.close()

#MONGO
client = MongoClient('localhost', 27017)
db = client.bigdata_ca2_elaine_chong
students = db.students
#incase repeat
result = students.delete_many({})
print("Delete result ", result.deleted_count)

#convert in python object
data = json_util.loads(json_data)
try:
#INSERT data into mongo
    students.insert_many(data)
except:
    print (e);

#create index
students.create_index('studentid')
def print_result(cursor):
    for document in cursor:
        pprint(document)
        print()

#RETURN result with id=45
cursor = students.find({"studentid": "45"})
print_result(cursor)
#cursor = students.find().limit(2)
#print_result(cursor)


#REDIS
r = redis.StrictRedis(host="localhost", port = 6379, db = 0)


#incase repeat
r.flushall()
#SET pipeline to speed up
pipe = r.pipeline()
json_dict  = json.load(open(json_path))

#INSERT into REDIS using hmset
for i in range(len(json_dict)):
    str = "student:"+json_dict[i]["studentid"]
    pipe.hmset(str, json_dict[i])

#EXECUTE call sends all buffered commands to the server, returning a list of responses, one for each command.
pipe.execute()
#RETURN student with id 1
pipe.hgetall("student:1")
print(pipe.execute())
#NEO4j
graph=Graph("http://localhost:7474/db/data/",user="neo4j", password="Neo4j")
graph.run("match (n:Student) delete n")
#INSERT DATA
graph.run("USING PERIODIC COMMIT 1000 LOAD CSV WITH HEADERS FROM 'http://elainepeilingchong.com/file/FinalProjectData.csv' AS line FIELDTERMINATOR ';' CREATE (p:Student {studentid:toInteger(line.studentid),lab_1: toInteger(line.Lab1), christmas_test: toInteger(line.ChristmasTest),lab_2: toInteger(line.Lab2),easter_test: toInteger(line.EasterTest), lab_3: toInteger(line.Lab3), part_time_job: toInteger(line.parttimejob), exam_grade: toInteger(line.ExamGrade)})").dump()
graph.run("CREATE INDEX ON :Student(studentid)")

#RETURN number of student
graph.run("MATCH (n:Student {}) RETURN count(*) as num").dump()

