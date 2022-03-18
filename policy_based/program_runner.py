import os
import time
from datetime import datetime
import numpy as np

# Set any unwished arguments (except gameName) to "-" if they are not desired, they will get default arguments in Go-Explore.
# gameName must be defined.
#####################
gameName           : str    = "maze"
loadPath           : str    = "-"
stepsPerIteration  : str    = "2000000"
levelSeed          : str    = "214"
testMode           : str    = "False"
posSeed            : str    = str(np.random.randint(0,623*1000)) # str(np.random.randint(0,623*1000)) <- This will give a random position with close to equal chance for any position.
maxTime            : str    = "65.0"
iterations         : int    = 1
#####################

#gameName           : What game to run.
#loadPath           : If a model is to be loaded initially, specify path here.
#stepsPerIteration  : The maximum steps per iteration.
#levelSeed          : Specify which level seed to run. str(np.random.randint(1,2147483648)) will give a random level amongst all possible seeds.
#testMode           : If the network should be freezed and tested or not.
#posSeed            : The seed to use when selecting starting position. str(np.random.randint(0,623*1000)) will give a random position with close to equal chance for any position.
#maxTime            : The maximum number of hours the program should run in total. Will be divided equally between iterations.
#iterations         : How many different iterations of the program should be run.


# Set this to the path to your 'temp' folder where the models from Go-Explore are stored.
# In ubuntu WSL it would probably be '/home/USERNAME/temp/'
# If 'iterations' is set to 1, this variable will not be used and does not have to be changed.
tempPath = '/home/nicolas/temp/' 


hoursPerIteration = str(float(maxTime)/iterations)


# This is used as a check such that the path in 'tempPath' exists.
if iterations > 1:
    os.listdir(tempPath)



#####################################################
###################STARTING RUNS#####################
#####################################################
logName = "program_runner_log.txt"

log  = open(logName, "w+")
log.write("Starting program_runner for: \ngameName=" + gameName + 
    "\nloadPath=" + loadPath + 
    "\nhoursPerLevel=" + hoursPerIteration + 
    "\nlevels=" + str(iterations) + 
    "\ndateAndTime=" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + 
    "\nlevelSeed="+ levelSeed +
    "\nposSeed=" + posSeed +
    "\ntestMode=" + testMode + 
    "\n\n")

startTime = time.time()

prevLoadPath = ""
errorHappened = False
nbrErrorHapppened = 0

print("Starting the first run out of " + str(iterations) + " runs at " + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

if iterations > 1:
    list = os.listdir(tempPath)
    list.sort()





for i in range(iterations):
    log.write("----------\n")
    tmp_string = str(round((iterations - i) * float(hoursPerIteration), 4)) + " hours remaining. Starting new run for loadPath=" + loadPath + "\n"
    log.write(tmp_string)
    log.write("levelSeed=" + levelSeed + "\n")
    log.write("posSeed=" + posSeed + "\n")
    print("############################################")
    print("############################################")
    print(tmp_string)
    log.flush()

    os.system("sh generic_atari_game.sh " + gameName + " " + loadPath + " " + hoursPerIteration + " " + stepsPerIteration + " " + levelSeed + " " + posSeed + " " + testMode)
    
    # Load the next run if this wasn't the last.
    if i < iterations - 1:
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

            tmp_string = "<<ERROR>>: Could not find any '.joblib' file at the location: " + loadPath + "\n"
            log.write(tmp_string)
            print(tmp_string)

            loadPath = prevLoadPath

            if nbrErrorHapppened > 4:
                tmp_string = str(nbrErrorHapppened) + " errors have occured, aborting run. Something is wrong."
                log.write(tmp_string)
                print(tmp_string)
                break

            tmp_string = "Using model from previous run instead located at: " + loadPath + "\n"
            log.write(tmp_string)
            print(tmp_string)
        posSeed = str(np.random.randint(0,623*1000))
    log.flush()

timeTaken = round((time.time() - startTime)/3600, 2)

if not errorHappened:
    tmp_string = "\n\nRun finished successfully at " + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + " with total time in hours of: " + str(timeTaken) + "."
else:
    tmp_string = "\n\n<<WARNING>>: Run finished at " + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + " with an error occuring."
log.write(tmp_string)
print(tmp_string + " Please see the file '" + logName + "' for more information.")

log.close()