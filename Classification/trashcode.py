import Image
from random import randint
def MapImg(Map):
   im=Image.new('RGB', (50, 50))
   Dict={"Woodlands":(255, 0, 0), "Plains":(255, 255, 255), "Waterlands":(0, 0, 255), "Grasslands":(0, 255, 0), "Rockylands":(0, 0, 0)}
   imMap=[]
   [[imMap.append(Dict[Env])forEnvinRow]forRowinMap]
   im.putdata(imMap)
   im.show()
   im.save('test.png')

   #img=[randint(0, 255)foryinrange(1)]#Tobereplacedwithactualpixels
   #print(img)
   #forpixel, colorinenumerate(img):
   #print(stylize("▮", colored.fg(color)), end='')
   #ifpixel%50==0andpixel!=0:
   #print('')

def ColPrint():
   #Author:DallinGuisti
   #PythonVersion:v3.7.1
   #ProgramVersion:1.0
   #Copyright(c)DallinGuisti2019.AllRightsReserved.
   defprintCol(text, color, end=False):
      Red='\033[31m'
      Green='\033[32m'
      Blue='\033[34m'
      White='\033[37m'
      Black='\033[30m'

   colors={"r":Red, "g":Green, "b":Blue, "w":White, "b":Black}
   ifcolorincolors.keys():
      printCol=colors[color]
   ifend!=False:
      print(printCol+text+White, end=end)
   else:
      print(printCol+text+White)

def DMapGen():
   classPixel():
      def__init__(self, biome, color):
         self.biome=biome
         self.color=color

def DMapPrint():
   pass#-----Ignoreallofthisifyoudon'twantcolorinyourgame-----
   ####print(Map)
   #whRat=2#Width-to-Heightratioofcharacterbeingusedtoprintmap
   #FMap=[Pixel("Tundra", 'g'), Pixel("Forest", 'g'), Pixel("Lake", 'b')]#Pleaseformatmapthiswaywithpixelclassesinordertobeabletonicelyreferenceinformationaboutanygivenpixelwhenneeded.Youcanmakethemapa2Darrayifyouwant, butpleasegenerateitoutofpixelsinordertoincludethebiomeinformation.Also, whenyoufinishthebiomes, pleaseincludetheassociatedcolorinthebiomeandwecanreferencethatforthewholebiomeratherthanstoringitinthepixel.
   #
   ##Onsecondthought, ifyouaregoingtohavebigchunksofland, itwouldprobablybemoreefficienttosimplystorethecoordinatesofpixelsincludedinagivenbiomeifyoucanthinkofawaytodoitthatway.It'suptoyou.
   #mapWidth=50
   #forindex, pixelinenumerate(FMap):
   #printCol("█", pixel.color, end='')
   #if(index+1)%(mapWidth*whRat)==0:
   #print('\n')

def NonStringClasses():
   pass
   """Forest=Biome("Soil", "Wood", "Leaves", "Bark", Fighter, Fighter, Fighter, Chicken, None, None)
   Jungle=Biome(Soil, Wood, Vines, Moss, Fighter, Fighter, Predator, Chicken, None, None)
   Grove=Biome(Soil, Wood, Fruit, Rocks, Fighter, Predator, Fighter, Chicken, None, None)
   Garden=Biome(Soil, Fruit, Gravel, Water, Fighter, Predator, Goblin, Chicken, Harvester's_Armor, Overlord)
   Desert=Biome("Sand", "Rocks", "Cacti", "Iron", Destroyer, Destroyer, Destroyer, Rabbit, None, None)
   Tundra=Biome(Snow, Rocks, Iron, Wood, Destroyer, Destroyer, Annihilator, Rabbit, None, None)
   Badlands=Biome(Sand, Snow, Bones, Diamonds, Destroyer, Annihilator, Destroyer, Rabbit, None, None)
   Temple=Biome(Sand, Iron, Wood, Gold, Destroyer, Annihilator, Troll, Rabbit, Strawman, Coffin)
   Prarie=Biome("Soil", "Gravel", "Stone", "Iron", Raider, Raider, Raider, Cow, None, None)
   Meadow=Biome(Soil, Flowers, Clay, Gold, Raider, Raider, Raider, Cow, None, None)
   Swamp=Biome(Mud, Water, Gold, Iron, Raider, Minion, Raider, Cow, None, None)
   Fort=Biome(Wood, Stone, Iron, Diamonds, Raider, Minion, Zombie, Cow, Ladder, Mob_Repellant)
   Lake=Biome(Water, Soil, Rocks, Emeralds, Defender, Defender, Defender, Fish, None, None)
   Beach=Biome(Water, Sand, Clay, Emeralds, Defender, Defender, Guardian, Fish, None, None)
   Island=Biome(Soil, Iron, Water, Quartz, Defender, Guardian, Defender, Fish, None, None)
   Shipwreck=Biome(Wood, Water, Gold, Quartz, Defender, Guardian, Skeleton, Fish, Bottled_Wave, Bottled_Wind)
   Mountain=Biome(Stone, Soil, Wood, Quartz, Hunter, Hunter, Hunter, Sheep, None, None)
   Canyon=Biome(Stone, Gravel, Leaves, Quartz, Hunter, Hunter, Assasin, Sheep, None, None)
   Cave=Biome(Stone, Iron, Wood, Quartz, Hunter, Assasin, Hunter, Sheep, None, None)
   Monument=Biome(Iron, Stone, Diamonds, Gravel, Hunter, Assasin, Ghoul, Sheep, Gliders, Binoculars)"""