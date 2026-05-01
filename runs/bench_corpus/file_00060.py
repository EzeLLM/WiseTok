import time

import MinecraftOCR
import constant
import numpy as np
import mysql.connector
import pyautogui
from datetime import datetime

from PIL import Image
from PIL import ImageGrab

# Initialize the OCR and a connection to the mysql server.
ocr = MinecraftOCR.OCR(Image.open('.\\data\\font.png'), 8, 8, 2)
mydb = mysql.connector.connect(
    host='10.0.0.29',
    user='laptop',
    password='@Mart',
    database='anniscraper'
)

mycursor = mydb.cursor()
# print(ocr.readErrorMessage(Image.open('.\\bad.png')))

matchNumber = 1
errorCount = 0

print("waiting...")
time.sleep(2)
print("done waiting")

while True:
    nextPlacing = 4
    placings = [5, 5, 5, 5]

    def joinGame():
        # Connect to the annihilation lobby
        pyautogui.click(button='right')
        time.sleep(1)
        pyautogui.moveTo(*constant.ANNI_LOBBY_LOCATION)
        pyautogui.click()
        time.sleep(1)
        pyautogui.moveTo(*constant.ANNI_HUB_ONE_LOCATION)
        pyautogui.click()
        time.sleep(3)

        # Continually scan through the servers until we find a game we want to join.
        currentCoords = constant.SERVERS_START_LOCATION
        gameFound = False
        while not gameFound:
            # Open up the server browser.
            pyautogui.click(button='right')
            time.sleep(0.5)

            for _ in range(9):
                pyautogui.moveTo(currentCoords)
                img = ImageGrab.grab()
                ocr.loadImage(img)
                phase = ocr.recognizeCharacter(*np.add(currentCoords, constant.SERVERS_PHASE_OFFSET), colors=[constant.GREEN])
                if phase == '1' or phase == '2':
                    gameFound = True
                    pyautogui.click()
                    break
                else:
                    currentCoords = tuple(np.add(currentCoords, constant.SERVERS_SPACING))
                    if currentCoords[0] > constant.SERVERS_END_LOCATION[0]:
                        currentCoords = constant.SERVERS_START_LOCATION

            if not gameFound:
                pyautogui.press('esc')
                time.sleep(30)

        # Wait 5 seconds before continuing to make sure the server is joined
        time.sleep(5)
        print('Server joined')


    def createMatch():
        initialImg = ImageGrab.grab()
        global matchNumber
        initialImg.save('matchStart%d.png' % matchNumber, 'PNG')
        matchNumber += 1
        ocr.loadImage(initialImg)

        matchID = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        mapName = ocr.recognizeMap()
        joinPhaseInfo = ocr.recognizePhase()

        print((matchID, mapName, *joinPhaseInfo))

        sql = "INSERT INTO matches (matchID, map, joinPhase, joinTime) VALUES (%s, %s, %s, %s)"
        mycursor.execute(sql, (matchID, mapName, *joinPhaseInfo))
        mydb.commit()

        print('Match created')
        return matchID


    def collectData():
        shotTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        img = ImageGrab.grab()
        img.save('lastGrab.png', 'PNG')
        # img = Image.open('.\\lastGrab.png')

        try:
            ocr.loadImage(img)
            disconnected = ocr.recognizeDisconnection()
            if disconnected:
                return True, disconnected

            health = ocr.recognizeHealth()
            damage = ocr.recognizeDamage()
            kills = ocr.recognizeKills()
            bossKill = ocr.recognizeBossKill()
            phase = ocr.recognizePhase()

            print(ocr.readName(978, 40, constant.WHITE))
            print(health)
            print(damage)
            print(kills)
            print(bossKill)
            print(phase)

            snapshotSQL = "INSERT INTO snapshots VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            mycursor.execute(snapshotSQL, (matchID, shotTime, *health, *damage, *bossKill, *phase))

            killsSQL = "INSERT INTO kills VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            for index in range(len(kills)):
                mycursor.execute(killsSQL, (matchID, shotTime, index, *kills[index]))
            mydb.commit()

            deadTeamCount = 0
            for teamIndex in range(4):
                if health[teamIndex] == ' 0' or health[teamIndex] == 0:
                    deadTeamCount += 1
                    if placings[teamIndex] == 5:
                        global nextPlacing
                        placings[teamIndex] = nextPlacing
                        nextPlacing -= 1

            print('Snapshot and kills added')
            return deadTeamCount >= 3, False
        except Exception as e:
            print(e)
            global errorCount
            imageName = 'error' + str(errorCount) + '.png'
            errorCount += 1
            print("Failed to process this image, saving as " + imageName)
            img.save(imageName, 'PNG')
            return False, False


    def backToLobby():
        img = ImageGrab.grab()
        img.save('lastGrab.png', 'PNG')
        ocr.loadImage(img)
        return ocr.recognizeLobby()


    def reconnect():
        pyautogui.click(*constant.BACK_TO_SERVERS_LOCATION)
        time.sleep(3600)
        pyautogui.click(*constant.SHOTBOW_LOCATION, clicks=2)

    # Find and join a game.
    joinGame()

    # Create the row for this match in the database
    matchID = createMatch()

    # The start collecting data
    matchDone = False
    crashed = False
    while not matchDone and not crashed:
        startTime = time.perf_counter()
        matchDone, crashed = collectData()
        processingTime = time.perf_counter() - startTime
        print('Took ' + str(processingTime))
        if processingTime < 0.5:
            time.sleep(0.5 - processingTime)

    if matchDone:
        for i in range(len(placings)):
            if placings[i] == 5:
                placings[i] = 1

        endMatchSQL = 'UPDATE matches ' \
                      'SET bluePlacing = %s, greenPlacing = %s, redPlacing = %s, yellowPlacing = %s ' \
                      'WHERE matchID = %s'
        mycursor.execute(endMatchSQL, (*placings, matchID))
        mydb.commit()
        # Wait until we get moved to the lobby before continuing
        while not backToLobby():
            time.sleep(0.5)
    else:
        reconnect()
