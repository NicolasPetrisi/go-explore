import os
import time
from datetime import datetime


# Set any unwished arguments (except gameName) to "-" if they are not desired.
#####################
gameName = "maze"
loadPath = "-"
hoursPerLevel = "4.0"
levels = 5
#####################


# Set this to the path to your 'temp' folder where the models are stored.
# In ubuntu WSL it would probably be '/home/USERNAME/temp/'
# If 'levels' it set to 1, this variable will not be used and does not have to be changed.
tempPath = '/home/nicolas/temp/' 



# This is used as a check such that the path above 'tempPath' is correct.
os.listdir(tempPath)



#####################################################
###################STARTING RUNS#####################
#####################################################
logName = "program_runner_log.txt"

log  = open(logName, "w+")
log.write("Starting program_runner for: \ngameName=" + gameName + 
    "\nloadPath=" + loadPath + 
    "\nhoursPerLevel=" + hoursPerLevel + 
    "\nlevels=" + str(levels) + 
    "\ndateAndTime=" + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + 
    "\n\n")

startTime = time.time()

prevLoadPath = ""
errorHappened = False
nbrErrorHapppened = 0

print("Starting the first run out of " + str(levels) + " runs at " + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

for i in range(levels):
    log.write(str(round((levels - i) * float(hoursPerLevel), 4)) + " hours remaining. Starting new run for loadPath=" + loadPath + "\n")
    
    os.system("sh generic_atari_game.sh " + gameName + " " + loadPath + " " + hoursPerLevel)
    
    # Load the next run if this wasn't the last.
    if i < levels - 1:
        prevLoadPath = loadPath
        loadPath = tempPath + os.listdir(tempPath)[-1] + "/"
        files = os.listdir(loadPath)

        modelFiles = []
        for file in files:
            if file.endswith(".joblib"):
                modelFiles.append(file)

        if len(modelFiles) > 0:
            loadPath = loadPath + "/" + modelFiles[-1]
        else:
            errorHappened = True
            nbrErrorHapppened += 1
            log.write("<<ERROR>>: Could not find any '.joblib' file at the location: " + loadPath + "\n")
            loadPath = prevLoadPath

            if nbrErrorHapppened > 2:
                log.write(str(nbrErrorHapppened) + " errors have occured, aborting run. Something is wrong.")
                break

            log.write("Using model from previous run instead located at: " + loadPath + "\n")

    log.write("----------\n")

timeTaken = round((time.time() - startTime)/3600, 2)

if not errorHappened:
    finalMessage = "\n\nRun finished successfully at " + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + " with total time in hours of: " + str(timeTaken) + "."
else:
    finalMessage = "\n\n<<WARNING>>: Run finished at " + str(datetime.now().strftime("%d/%m/%Y %H:%M:%S")) + " with an error occuring."
log.write(finalMessage)
print(finalMessage + " Please see the file '" + logName + "' for more information.")

log.close()

