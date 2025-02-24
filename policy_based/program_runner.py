from argparse import ArgumentError
import os
from datetime import datetime
from datetime import timedelta
import numpy as np
from signal import SIGINT

# Set any unwished arguments (except gameName) to "-" if they are not desired,
# they will get default arguments in Go-Explore. If an argument does not have standard value then a warning will be raised.

# NOTE: If the program would have too little execution time, such as setting max steps or time to 0 the program will crash with
# a "referenced before assigned" or "index out of range" error. This is because the program will start but then terminate before
# it has had a chance to write the first checkpoint. Then during exit it will try to access values it should have created
# during runtime but not had time to do. This is why 0.1 is the minimum time allowed. There is however no safety min for max frames.
#####################
gameName           : str    = "maze"                # maze
minimumIterations  : int    = 1                     # For training: hours / 10. For test: 1
levelSeed          : str    = "137"                 # 137

posSeed            : str    = "-5"                  # For training: "-1". For test: "-5".

testMode           : bool   = False                  # For training: False. For test: True
endTime            : str    = "1.0"                   # For training: "72". For test: long enough
tempPath           : str    = '/home/nicolas/temp/' # '/home/nicolasfredrik/temp/'
folder             : str    = "-"                   # "-" 
loadPathModel      : str    = "-"                   # "-"
loadPathArch       : str    = "-"                   # "-"
loadPathTrajectory : str    = "-"                   # "-"
stepsPerIteration  : str    = "20000000"            # "20000000"
numberOfCores      : str    = "1"                   # For training: "8". For test: "1"
videoAllEpisodes   : str    = "True"                # For training: "False". For test: "True"
continue_run       : bool   = False                 # For training: True. For test: False
#####################

#gameName           : What game to run.
#minimumIterations  : How many different iterations of the program should at least be run. If given time due to early stopping, more iterations could occure.
#levelSeed          : Specify which level seed to run. str(np.random.randint(1,2147483648)) will give a random level amongst all possible seeds.
#posSeed            : The seed to use when selecting starting position. 0 will always be bottom left corner. str(np.random.randint(0,623*1000)) will give a random position with close to equal chance for any position.
#testMode           : If the network should be freezed and tested or not.
#endTime            : At what time and date the run should be finished (YYYY-mm-dd HH:MM:SS). Alternativelly set to number of hours as a float (X). The time will be divided equally between iterations.
#tempPath           : Set this to the path to your 'temp' folder where the models from Go-Explore are stored. In ubuntu WSL it would probably be '/home/USERNAME/temp/' 
#folder             : Which folder in tempPath that the model, archive and trajectory file resides.
#loadPathModel      : If a model is to be loaded initially, specify path from tempPath here.
#loadPathArch       : The archive to load initially, specify path from tempPath here.
#stepsPerIteration  : The maximum steps per iteration.
#numberOfCores      : How many cores of the CPU to use during the run.
#videoAllEpisodes   : If True, then a video for every episodes will be made, if False only every min(2^N, 500) video will be made.
#continue_run       : If True, then Go-Explore will automatically load the most recent run saved in the temp folder when it restarts. For example when minimumIterations is greater than 1.




list = os.listdir(tempPath)
list.sort()


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
    "\nfolder=" + folder + 
    "\nloadPathModel=" + loadPathModel + 
    "\nloadPathArch=" + loadPathArch + 
    "\nhoursPerLevel=" + hoursPerIteration + 
    "\nlevels=" + str(minimumIterations) + 
    "\nstartDate=" + str(startTime) + 
    "\nendDate=" + str(endTime) +
    "\nlevelSeed="+ levelSeed +
    "\nposSeed=" + posSeed +
    "\ntestMode=" + str(testMode) + 
    "\n\n")


prevLoadPath = ""
errorsHapppened = []
minTimeAllowed = 0.1

print("Starting the first run out of " + str(minimumIterations) + " runs at " + str(datetime.now().strftime(format)))   
print("Expected to be finished at", str(endTime)) 
print("Making video for every episode:", videoAllEpisodes)
first_run = True

while datetime.now() < datetime.strptime(endTime, format):

    remaining_time = round((datetime.strptime(endTime, format) - datetime.now()).total_seconds()/3600, 2)
    if remaining_time < float(hoursPerIteration):
        if remaining_time < minTimeAllowed:
            break
        hoursPerIteration = str(round(remaining_time, 2))

    log.write("----------\n")
    tmpString = str(remaining_time) + " hours remaining. Starting new run from folder: " + folder + " with loadPathModel=" + loadPathModel + " and loadPathArch=" + loadPathArch + " of " + hoursPerIteration + " hours.\n"
    log.write(tmpString)
    log.write("levelSeed=" + levelSeed + "\n")
    log.write("posSeed=" + posSeed + "\n")
    print("############################################")
    print("############################################")
    print(tmpString)
    log.flush()

    cont = "-"
    if continue_run and not first_run:
        cont = "True"

    returnValue = os.system("sh run_procgen_game.sh " + gameName + " " + loadPathModel + " " + loadPathArch + " " + hoursPerIteration + " " + stepsPerIteration + " " + levelSeed + " " + posSeed + " " + str(testMode) + " " + videoAllEpisodes + " " + numberOfCores + " " + folder + " " + loadPathTrajectory + " " + cont)

    
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
        if not continue_run:
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
    else:
        log.write("Only " + str(remaining_time) + " hours left but minimum time allowed is " + str(minTimeAllowed) + ". Stopping.\n")
    if len(errorsHapppened) > 0:
        print("<<WARNING>> More than 4 errors have occured. Something is wrong, stopping program.")
        break
        
    first_run = False
    if continue_run:
        loadPathArch = "-"
        loadPathModel = "-"
        loadPathTrajectory = "-"
        folder = "-"
    log.flush()

    if testMode:
        break

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