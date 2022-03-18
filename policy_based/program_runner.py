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
minimumIterations  : int    = 5
levelSeed          : str    = "214"
posSeed            : str    = str(np.random.randint(0,623*1000))
testMode           : str    = "False"
endTime            : str    = "2022-03-21 09:00:00"
tempPath           : str    = '/home/nicolasfredrik/temp/'
loadPath           : str    = "-"
stepsPerIteration  : str    = "20000000"
#####################

#gameName           : What game to run.
#minimumIterations  : How many different iterations of the program should at least be run. If given time due to early stopping, more iterations could occure.
#levelSeed          : Specify which level seed to run. str(np.random.randint(1,2147483648)) will give a random level amongst all possible seeds.
#posSeed            : The seed to use when selecting starting position. str(np.random.randint(0,623*1000)) will give a random position with close to equal chance for any position.
#testMode           : If the network should be freezed and tested or not.
#endTime            : At what time and date the run should be finished (YYYY-mm-dd HH:MM:SS). Alternativelly set to number of hours as a float (X). The time will be divided equally between iterations.
#tempPath           : Set this to the path to your 'temp' folder where the models from Go-Explore are stored. In ubuntu WSL it would probably be '/home/USERNAME/temp/' 
#loadPath           : If a model is to be loaded initially, specify path here.
#stepsPerIteration  : The maximum steps per iteration.

list = os.listdir(tempPath)
list.sort()

format = "%Y-%m-%d %H:%M:%S"
startTime = datetime.now().strftime(format)

if datetime.strptime(endTime, format) < datetime.strptime(startTime, format):
    raise Exception("End time can not be set in the past. Must be forward in time.")


try:
    maxHours = float(endTime)
    endTime = (datetime.now() + timedelta(seconds=int(float(endTime)*3600))).strftime(format)
except:
    maxTime = datetime.strptime(endTime, format) - datetime.strptime(startTime, format)
    maxHours = float(maxTime.total_seconds()/3600)

hoursPerIteration = str(maxHours/minimumIterations)



#####################################################
###################STARTING RUNS#####################
#####################################################
logName = "program_runner_log.txt"

log  = open(logName, "w+")
log.write("Starting program_runner for: \ngameName=" + gameName + 
    "\nloadPath=" + loadPath + 
    "\nhoursPerLevel=" + hoursPerIteration + 
    "\nlevels=" + str(minimumIterations) + 
    "\nstartDate=" + str(startTime) + 
    "\nendDate=" + str(endTime) +
    "\nlevelSeed="+ levelSeed +
    "\nposSeed=" + posSeed +
    "\ntestMode=" + testMode + 
    "\n\n")


prevLoadPath = ""
errorHappened = False
nbrErrorHapppened = 0
minTimeAllowed = 0.05

print("Starting the first run out of " + str(minimumIterations) + " runs at " + str(datetime.now().strftime(format)))    

while datetime.now() < datetime.strptime(endTime, format):

    remaining_time = (datetime.strptime(endTime, format) - datetime.now()).total_seconds()/3600
    if remaining_time < float(hoursPerIteration):
        if remaining_time < minTimeAllowed:
            break
        hoursPerIteration = str(round(remaining_time, 2))

    log.write("----------\n")
    tmpString = str(round(remaining_time, 2)) + " hours remaining. Starting new run for loadPath=" + loadPath + " of " + hoursPerIteration + " hours.\n"
    log.write(tmpString)
    log.write("levelSeed=" + levelSeed + "\n")
    log.write("posSeed=" + posSeed + "\n")
    print("############################################")
    print("############################################")
    print(tmpString)
    log.flush()

    returnValue = os.system("sh generic_atari_game.sh " + gameName + " " + loadPath + " " + hoursPerIteration + " " + stepsPerIteration + " " + levelSeed + " " + posSeed + " " + testMode)

    # If we got keyboard interrupt, then break the loop to exit the program.
    if returnValue == SIGINT:
        tmpString = "\n<<ERROR>>: Got keyboard interrupt, stopping program.\n\n"
        errorHappened = True
        nbrErrorHapppened += 1
        print(tmpString)
        log.write(tmpString)
        break

    # Load the next run if this wasn't the last.
    remaining_time = (datetime.strptime(endTime, format) - datetime.now()).total_seconds()/3600
    if remaining_time > minTimeAllowed:
        prevLoadPath = loadPath
        list = os.listdir(tempPath)
        list.sort()
        loadPath = tempPath + list[-1] + "/"


        files = os.listdir(loadPath)


        modelFiles = []
        for file in files:
            if file.endswith(".joblib"):
                modelFiles.append(file)

        if len(modelFiles) > 0:
            loadPath = loadPath + modelFiles[-1]
        else:
            errorHappened = True
            nbrErrorHapppened += 1

            tmpString = "<<ERROR>>: Could not find any '.joblib' file at the location: " + loadPath + "\n"
            log.write(tmpString)
            print(tmpString)

            loadPath = prevLoadPath

            if nbrErrorHapppened > 4:
                tmpString = str(nbrErrorHapppened) + " errors have occured, aborting run. Something is wrong."
                log.write(tmpString)
                print(tmpString)
                break

            tmpString = "Using model from previous run instead located at: " + loadPath + "\n"
            log.write(tmpString)
            print(tmpString)
        posSeed = str(np.random.randint(0,623*1000))

    log.flush()

timeTaken = datetime.strptime(datetime.now().strftime(format), format) - datetime.strptime(startTime, format)


if not errorHappened:
    tmpString = "\n\nRun finished successfully at " + str(datetime.now().strftime(format)) + " with total time of: " + str(timeTaken) + "."
else:
    tmpString = "\n\n<<WARNING>>: Run finished at " + str(datetime.now().strftime(format)) + " with an error occuring."
log.write(tmpString)
print(tmpString + " Please see the file '" + logName + "' for more information.")

log.close()