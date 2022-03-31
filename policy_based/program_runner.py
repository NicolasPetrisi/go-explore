import os
import time
from datetime import datetime
from datetime import timedelta
import numpy as np
from signal import SIGINT, siginterrupt

# Set any unwished arguments (except gameName) to "-" if they are not desired,
# they will get default arguments in Go-Explore.
# gameName, tempPath and endTime MUST be defined.
#####################
gameName           : str    = "maze"
minimumIterations  : int    = 1
levelSeed          : str    = "92"
posSeed            : str    = "0"
testMode           : str    = "False"
endTime            : str    = "0.1"
tempPath           : str    = '/home/nicolas/temp/'
loadPathModel      : str    = "2363215_452fe0866c354aa49d6690cb095280a0/000000010240_model.joblib"
loadPathArch       : str    = "2363215_452fe0866c354aa49d6690cb095280a0/000000010240_arch.gz"
stepsPerIteration  : str    = "20000000"
numberOfCores      : str    = "1"
videoAllEpisodes   : str    = "False"
#####################

#gameName           : What game to run.
#minimumIterations  : How many different iterations of the program should at least be run. If given time due to early stopping, more iterations could occure.
#levelSeed          : Specify which level seed to run. str(np.random.randint(1,2147483648)) will give a random level amongst all possible seeds.
#posSeed            : The seed to use when selecting starting position. 0 will always be bottom left corner. str(np.random.randint(0,623*1000)) will give a random position with close to equal chance for any position.
#testMode           : If the network should be freezed and tested or not.
#endTime            : At what time and date the run should be finished (YYYY-mm-dd HH:MM:SS). Alternativelly set to number of hours as a float (X). The time will be divided equally between iterations.
#tempPath           : Set this to the path to your 'temp' folder where the models from Go-Explore are stored. In ubuntu WSL it would probably be '/home/USERNAME/temp/' 
#loadPathModel      : If a model is to be loaded initially, specify path from tempPath here.
#loadPathArch       : The archive to load initially, specify path from tempPath here.
#stepsPerIteration  : The maximum steps per iteration.
#numberOfCores      : How many cores of the CPU to use during the run.
#videoAllEpisodes   : If True, then a video for every episodes will be made, if False only every min(2^N, 500) video will be made.

list = os.listdir(tempPath)
list.sort()

if loadPathModel != "-":
    loadPathModel = tempPath + loadPathModel

if loadPathArch != "-":
    loadPathArch = tempPath + loadPathArch

format = "%Y-%m-%d %H:%M:%S"
startTime = datetime.now().strftime(format)



try:
    maxHours = float(endTime)
    endTime = (datetime.now() + timedelta(seconds=int(float(endTime)*3600))).strftime(format)
except:
    if datetime.strptime(endTime, format) < datetime.strptime(startTime, format):
        raise Exception("End time can not be set in the past. Must be forward in time.")

    maxTime = datetime.strptime(endTime, format) - datetime.strptime(startTime, format)
    maxHours = float(maxTime.total_seconds()/3600)

hoursPerIteration = str(round(maxHours/minimumIterations, 2))



#####################################################
###################STARTING RUNS#####################
#####################################################
logName = "program_runner_log.txt"

log  = open(logName, "w+")
log.write("Starting program_runner for: \ngameName=" + gameName + 
    "\nloadPathModel=" + loadPathModel + 
    "\nloadPathArch=" + loadPathArch + 
    "\nhoursPerLevel=" + hoursPerIteration + 
    "\nlevels=" + str(minimumIterations) + 
    "\nstartDate=" + str(startTime) + 
    "\nendDate=" + str(endTime) +
    "\nlevelSeed="+ levelSeed +
    "\nposSeed=" + posSeed +
    "\ntestMode=" + testMode + 
    "\n\n")


prevLoadPath = ""
errorsHapppened = []
minTimeAllowed = 0.1

print("Starting the first run out of " + str(minimumIterations) + " runs at " + str(datetime.now().strftime(format)))   
print("Expected to be finished at", str(endTime)) 
print("Making video for every episode:", videoAllEpisodes)

while datetime.now() < datetime.strptime(endTime, format):

    remaining_time = round((datetime.strptime(endTime, format) - datetime.now()).total_seconds()/3600, 2)
    if remaining_time < float(hoursPerIteration):
        if remaining_time < minTimeAllowed:
            break
        hoursPerIteration = str(round(remaining_time, 2))

    log.write("----------\n")
    tmpString = str(remaining_time) + " hours remaining. Starting new run for loadPathModel=" + loadPathModel + " and loadPathArch=" + loadPathArch + " of " + hoursPerIteration + " hours.\n"
    log.write(tmpString)
    log.write("levelSeed=" + levelSeed + "\n")
    log.write("posSeed=" + posSeed + "\n")
    print("############################################")
    print("############################################")
    print(tmpString)
    log.flush()

    returnValue = os.system("sh generic_atari_game.sh " + gameName + " " + loadPathModel + " " + loadPathArch + " " + hoursPerIteration + " " + stepsPerIteration + " " + levelSeed + " " + posSeed + " " + testMode + " " + videoAllEpisodes + " " + numberOfCores)

    
    if returnValue != 0:
        # If we got keyboard interrupt; break the loop to exit the program.
        if returnValue == SIGINT:
            tmpString = "\n<<INTERRUPT>>: Got keyboard interrupt, stopping program.\n\n"
            errorsHapppened.append("Error code [" + str(returnValue) + "]")
            print(tmpString)
            log.write(tmpString)
            break
        
        #Otherwise something else went wrong.
        tmpString = "\n<<ERROR>>: Program returned error code: [" + str(returnValue) + "].\n\n"
        errorsHapppened.append(("Error code [" + str(returnValue) + "]"))
        print(tmpString)
        log.write(tmpString)

    # Load the next run if this wasn't the last.
    remaining_time = (datetime.strptime(endTime, format) - datetime.now()).total_seconds()/3600
    if remaining_time > minTimeAllowed:
        prevLoadPath = loadPathModel
        list = os.listdir(tempPath)
        list.sort()
        loadPathModel = tempPath + list[-1] + "/"


        files = os.listdir(loadPathModel)


        modelFiles = []
        for file in files:
            if file.endswith(".joblib"):
                modelFiles.append(file)

        if len(modelFiles) > 0:
            loadPathModel = loadPathModel + modelFiles[-1]
        else:
            errorsHapppened.append("no .joblib file found")

            tmpString = "<<ERROR>>: Could not find any '.joblib' file at the location: " + loadPathModel + "\n"
            log.write(tmpString)
            print(tmpString)

            loadPathModel = prevLoadPath

            tmpString = "Using model from previous run instead located at: " + loadPathModel + "\n"
            log.write(tmpString)
            print(tmpString)
        posSeed = str(np.random.randint(0,623*1000))

    if len(errorsHapppened) > 4:
        print("<<WARNING>> More than 4 errors have occured. Something is wrong, stopping program.")
        break
        

    log.flush()

timeTaken = datetime.strptime(datetime.now().strftime(format), format) - datetime.strptime(startTime, format)


if not errorsHapppened:
    tmpString = "\n\nRun finished successfully at " + str(datetime.now().strftime(format)) + " with total time of: " + str(timeTaken) + "."
else:
    tmpString = "<<WARNING>> " + str(errorsHapppened) + " errors have occured."
    log.write(tmpString)
    print(tmpString)
    tmpString = "\n\n<<WARNING>>: Run finished at " + str(datetime.now().strftime(format)) + " with " + str(len(errorsHapppened)) + " errors occuring."
log.write(tmpString)
print(tmpString + " Please see the file '" + logName + "' for more information.")

log.close()