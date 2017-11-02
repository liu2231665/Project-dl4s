"""#########################################################################
Author: Yingru Liu
Institute: Stony Brook University
Descriptions: the files contain the tools to extract the piano-rolls
              representations of the midi files and to save them as
              hdf5 data-set. The Piano-rolls representation is binary
              matrix for each segment and the relationships of the
              instruments are neglected.
              ----2017.11.01
#########################################################################"""

import os
import urllib.request
import tarfile
import h5py
import pretty_midi
import numpy as np

# Data name.URL.
Lakh_HDF5 = "./dataset/Lakh_clean.hdf5"
Lakh_RAW = "./dataset/clean_midi.tar.gz"
Lakh_URL = "http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz"

TRAIN_RATIO = 0.9
Valid_RATIO = 0.95

"""#########################################################################
There are some files that can not be read by the prettyMidi.
The reason is unknown.
#########################################################################"""
FAILLIST = [
    './dataset/clean_midi/3T/Why.mid',
    './dataset/clean_midi/10cc/Dreadlock Holiday.4.mid',
    "./dataset/clean_midi/4 Non Blondes/What's Up.5.mid",
    "./dataset/clean_midi/a-ha/Take On Me.1.mid",
    "./dataset/clean_midi/Aaron Neville/Tell It Like It Is.mid",
    "./dataset/clean_midi/ABBA/I've Been Waiting For You.mid",
    "./dataset/clean_midi/ABBA/One Of Us.mid",
    "./dataset/clean_midi/ABBA/Take a Chance on Me.3.mid",
    "./dataset/clean_midi/ABBA/Thank You for the Music.2.mid",
    "./dataset/clean_midi/ABBA/Voulez Vous.1.mid",
    "./dataset/clean_midi/Aerosmith/Dream On.mid",
    "./dataset/clean_midi/Aerosmith/Pink.3.mid",
    "./dataset/clean_midi/Alice in Chains/Sludge Factory.mid",
    "./dataset/clean_midi/Alison Moyet/All Cried Out.mid",
    "./dataset/clean_midi/Amedeo Minghi/Decenni.1.mid",
    "./dataset/clean_midi/Amos, Tori/The Wrong Band.mid",
    "./dataset/clean_midi/Andre Brasseur/Early Bird Satellite.1.mid",
    "./dataset/clean_midi/Andre Hazes/Ik meen 't.2.mid",
    "./dataset/clean_midi/Aqua/Dr Jones.mid",
    "./dataset/clean_midi/Asia/Don't Cry.1.mid",
    "./dataset/clean_midi/Barry Manilow/Copa Cabana (disco).mid",
    "./dataset/clean_midi/Basie/I Left My Heart in San Francisco.mid",
    "./dataset/clean_midi/Beastie Boys/Girls.mid",
    "./dataset/clean_midi/Bee Gees/How Can You Mend a Broken Heart.1.mid",
    "./dataset/clean_midi/Bee Gees/Stayin Alive.3.mid",
    "./dataset/clean_midi/Bee Gees/Too Much Heaven.mid",
    "./dataset/clean_midi/Bernstein Leonard/West Side Story: Medley.1.mid",
    "./dataset/clean_midi/Billy Joel/Movin' Out (Anthony's Song).mid",
    "./dataset/clean_midi/Billy Joel/Pressure.2.mid",
    "./dataset/clean_midi/Billy Joel/Pressure.mid",
    "./dataset/clean_midi/Billy Swan/I Can Help.1.mid",
    "./dataset/clean_midi/Blondie/Dreaming.mid",
    "./dataset/clean_midi/Bon Jovi/Blaze of Glory.2.mid",
    "./dataset/clean_midi/Boyz II Men/End of the Road.2.mid",
    "./dataset/clean_midi/Brian McKnight/On the Down Low.mid",
    "./dataset/clean_midi/Bruce Springsteen/Tenth Avenue Freeze-Out.1.mid",
    "./dataset/clean_midi/Bryan Adams/(Everything I Do) I Do It For You.7.mid",
    "./dataset/clean_midi/Bryson/Tonight I Celebrate My Love For You.mid",
    "./dataset/clean_midi/Buddy Holly/Peggy Sue.1.mid",
    "./dataset/clean_midi/Bush/Glycerine.1.mid",
    "./dataset/clean_midi/Busta Rhymes/Woo Hah.mid",
    "./dataset/clean_midi/Cabrel Francis/Encore et encore.mid",
    "./dataset/clean_midi/Cabrel Francis/Question d'equilibre.mid",
    "./dataset/clean_midi/Camel/The Snow Goose.1.mid",
    "./dataset/clean_midi/Camel/The Snow Goose.mid",
    "./dataset/clean_midi/Celine Dion/That's The Way It Is.mid",
    "./dataset/clean_midi/Celine Dion/Where Does My Heart Beat Now.1.mid",
    "./dataset/clean_midi/Celine Dion/Where Does My Heart Beat Now.mid",
    "./dataset/clean_midi/Chic/Chic Mystique.mid",
    "./dataset/clean_midi/Chic/Take My Love.mid",
    "./dataset/clean_midi/Clayderman Richard/Ballade Pour Adeline.mid",
    "./dataset/clean_midi/Cocciante/Ammassati e distanti.mid",
    "./dataset/clean_midi/Commodores/Three Times a Lady.1.mid",
    "./dataset/clean_midi/Coolio/Gangsta's Paradise.3.mid",
    "./dataset/clean_midi/Cream/Strange Brew.1.mid",
    "./dataset/clean_midi/Crow Sheryl/Tomorrow Never Dies.mid",
    "./dataset/clean_midi/Daft Punk/Da Funk.mid",
    "./dataset/clean_midi/Dalla/Anna e Marco.mid",
    "./dataset/clean_midi/Dan Fogelberg/Leader Of The Band.mid",
    "./dataset/clean_midi/Darin Bobby/Splish Splash.mid",
    "./dataset/clean_midi/Dee/The Legend Of Xanadu.1.mid",
    "./dataset/clean_midi/Depeche Mode/Shake the Disease.mid",
    "./dataset/clean_midi/Earth, Wind & Fire/September (bonus track).4.mid",
    "./dataset/clean_midi/Elton John/A Word in Spanish.1.mid",
    "./dataset/clean_midi/Elton John/Nikita.mid",
    "./dataset/clean_midi/Emerson, Lake & Palmer/Fanfare for the Common Man.2.mid",
    "./dataset/clean_midi/Emerson, Lake & Palmer/Hoedown.mid",
    "./dataset/clean_midi/Energy 52/Cafe del Mar.mid",
    "./dataset/clean_midi/Enya/Bard Dance.mid",
    "./dataset/clean_midi/Eric Clapton/Tears in Heaven.7.mid",
    "./dataset/clean_midi/Eurythmics/Sweet Dreams.4.mid",
    "./dataset/clean_midi/Eurythmics/Who's That Girl.mid",
    "./dataset/clean_midi/Frank Sinatra/Summer Wind.mid",
    "./dataset/clean_midi/Garbage/Cherry Lips.mid",
    "./dataset/clean_midi/Garbage/Stupid Girl.3.mid",
    "./dataset/clean_midi/Genesis/Abacab.3.mid",
    "./dataset/clean_midi/Genesis/Misunderstanding.3.mid",
    "./dataset/clean_midi/Gina G/Ooh Ahh Just a Little Bit.mid",
    "./dataset/clean_midi/Gompie/Alice (Who the X Is Alice) (Living Next Door to Alice).mid",
    "./dataset/clean_midi/Henry Arland/Rosenmelodie.mid",
    "./dataset/clean_midi/Henry Mancini/Moon River.mid",
    "./dataset/clean_midi/Huey Lewis & The News/The Power of Love.mid",
    "./dataset/clean_midi/Jackson Michael/Childhood.mid",
    "./dataset/clean_midi/Jackson Michael/Don't Stop 'Til You Get Enough.mid",
    "./dataset/clean_midi/Jackson Michael/I'll Be There.mid",
    "./dataset/clean_midi/Jackson Michael/Smooth Criminal.4.mid",
    "./dataset/clean_midi/Jackson Michael/Smooth Criminal.mid",
    "./dataset/clean_midi/Jackson Michael/Thriller.3.mid",
    "./dataset/clean_midi/Jackson Michael/You Are Not Alone.mid",
    "./dataset/clean_midi/Jamiroquai/Canned Heat.1.mid",
    "./dataset/clean_midi/Jean Michel Jarre/Calypso, Part 2.mid",
    "./dataset/clean_midi/Jennifer Lopez/If You Had My Love.mid",
    "./dataset/clean_midi/Jethro Tull/Rainbow Blues.mid",
    "./dataset/clean_midi/Jimi Hendrix/Purple Haze.2.mid",
    "./dataset/clean_midi/John Elton/Nikita.2.mid",
    "./dataset/clean_midi/John Paul Young/Love is in the Air.3.mid",
    "./dataset/clean_midi/Johnny Mercer/Come Rain or Come Shine.mid",
    "./dataset/clean_midi/Journey/Any Way You Want It.mid",
    "./dataset/clean_midi/Journey/Lights.mid",
    "./dataset/clean_midi/Journey/Lovin' Touchin' Squeezin'.mid",
    "./dataset/clean_midi/Jovanotti/Penso Positivo.mid",
    "./dataset/clean_midi/Juan Luis Guerra/La Bilirrubina.1.mid",
    "./dataset/clean_midi/Kalua Beach Boys/There's No Place Like Hawaii.mid",
    "./dataset/clean_midi/Kylie Minogue/Better the Devil.mid",
    "./dataset/clean_midi/Kylie Minogue/Especially for You.1.mid",
    "./dataset/clean_midi/Last/Rosamunde.1.mid",
    "./dataset/clean_midi/Led Zeppelin/Custard Pie.mid",
    "./dataset/clean_midi/Led Zeppelin/In The Evening.mid",
    "./dataset/clean_midi/Led Zeppelin/Stairway To Heaven.mid",
    "./dataset/clean_midi/Lenny Kravitz/I Belong To You.mid",
    "./dataset/clean_midi/Level 42/Lessons in Love.2.mid",
    "./dataset/clean_midi/Live/Selling the Drama.mid",
    "./dataset/clean_midi/LL Cool J/Hey Lover.mid",
    "./dataset/clean_midi/Los Del Rio/Macarena.1.mid",
    "./dataset/clean_midi/Los Lobos/La Bamba.mid",
    "./dataset/clean_midi/Madonna/Deeper and Deeper.mid",
    "./dataset/clean_midi/Madonna/Like A Virgin.mid",
    "./dataset/clean_midi/Madonna/Rain.4.mid",
    "./dataset/clean_midi/Marc Anthony/Vivir Lo Nuestro.mid",
    "./dataset/clean_midi/Mariah Carey/Forever.mid",
    "./dataset/clean_midi/Mariah Carey/Long Ago.mid",
    "./dataset/clean_midi/Marley Bob/Iron Lion Zion.mid",
    "./dataset/clean_midi/Matt Bianco/Half a Minute.mid",
    "./dataset/clean_midi/McDonald/Yah Mo B There.mid",
    "./dataset/clean_midi/Metallica/(Anesthesia)-Pulling Teeth.mid",
    "./dataset/clean_midi/Metallica/The Shortest Straw.1.mid",
    "./dataset/clean_midi/Metallica/The Unforgiven.3.mid",
    "./dataset/clean_midi/Metallica/The Unforgiven.5.mid",
    "./dataset/clean_midi/Miami Sound Machine/Here We Are.mid",
    "./dataset/clean_midi/Miller/When Sunny Gets Blue.mid",
    "./dataset/clean_midi/Morton Jelly Roll/Frog-I-More Rag.mid",
    "./dataset/clean_midi/Neil Diamond/(Encore) Cracklin' Rose.mid",
    "./dataset/clean_midi/Neil Diamond/Hello Again.mid",
    "./dataset/clean_midi/Nek/Se io non avessi te.2.mid",
    "./dataset/clean_midi/Nelly/Country Grammar (Hot ...).mid",
    "./dataset/clean_midi/Nick Kamen/I Promised Myself.1.mid",
    "./dataset/clean_midi/Nini Rosso/Il Silenzio.1.mid",
    "./dataset/clean_midi/Nirvana/(New Wave) Polly.mid",
    "./dataset/clean_midi/Nirvana/All Apologies.1.mid",
    "./dataset/clean_midi/Nirvana/Aneurysm.mid",
    "./dataset/clean_midi/Nirvana/Been a Son.mid",
    "./dataset/clean_midi/Nirvana/Dive.mid",
    "./dataset/clean_midi/Nirvana/Downer.1.mid",
    "./dataset/clean_midi/Nirvana/Drain You.mid",
    "./dataset/clean_midi/Nirvana/Dumb.2.mid",
    "./dataset/clean_midi/Nirvana/Frances Farmer Will Have Her Revenge on Seattle.1.mid",
    "./dataset/clean_midi/Nirvana/Hairspray Queen.1.mid",
    "./dataset/clean_midi/Nirvana/Heart-Shaped Box.mid",
    "./dataset/clean_midi/Nirvana/Lounge Act.mid",
    "./dataset/clean_midi/Nirvana/Love Buzz.mid",
    "./dataset/clean_midi/Nirvana/Mexican Seafood.mid",
    "./dataset/clean_midi/Nirvana/Milk It.1.mid",
    "./dataset/clean_midi/Nirvana/Mr. Moustache.mid",
    "./dataset/clean_midi/Nirvana/Negative Creep.mid",
    "./dataset/clean_midi/Nirvana/Pennyroyal Tea.2.mid",
    "./dataset/clean_midi/Nirvana/Polly.1.mid",
    "./dataset/clean_midi/Nirvana/Radio Friendly Unit Shifter.1.mid",
    "./dataset/clean_midi/Nirvana/Scentless Apprentice.mid",
    "./dataset/clean_midi/Nirvana/Sliver.2.mid",
    "./dataset/clean_midi/Nirvana/Smells Like Teen Spirit.8.mid",
    "./dataset/clean_midi/Nirvana/Son of a Gun.2.mid",
    "./dataset/clean_midi/Nirvana/Swap Meet.mid",
    "./dataset/clean_midi/Nirvana/Territorial Pissings.mid",
    "./dataset/clean_midi/Nirvana/tourette's.1.mid",
    "./dataset/clean_midi/Nirvana/Turnaround.mid",
    "./dataset/clean_midi/Nomadi/Il vento del nord.mid",
    "./dataset/clean_midi/Nomadi/Io vagabondo.2.mid",
    "./dataset/clean_midi/Nomadi/Io vagabondo.4.mid",
    "./dataset/clean_midi/Parker Charlie/Donna Lee.mid",
    "./dataset/clean_midi/Pet Shop Boys/Always on My Mind   In My House.3.mid",
    "./dataset/clean_midi/Peter Gabriel/Solsbury Hill.1.mid",
    "./dataset/clean_midi/Peter Gabriel/Solsbury Hill.mid",
    "./dataset/clean_midi/Peter, Paul & Mary/Puff.1.mid",
    "./dataset/clean_midi/Phish/Taste.mid",
    "./dataset/clean_midi/Pink Floyd/Echoes.1.mid",
    "./dataset/clean_midi/Pink Floyd/Echoes.mid",
    "./dataset/clean_midi/Pink Floyd/Hey You.mid",
    "./dataset/clean_midi/Pink Floyd/High Hopes.2.mid",
    "./dataset/clean_midi/Pink Floyd/Nobody Home.mid",
    "./dataset/clean_midi/Pink Floyd/The Trial.1.mid",
    "./dataset/clean_midi/Pink Floyd/Wish You Were Here.mid",
    "./dataset/clean_midi/Pooh/Il cielo e blu sopra le nuvole.mid",
    "./dataset/clean_midi/Prince/Let's Go Crazy.3.mid",
    "./dataset/clean_midi/R.E.M./Nightswimming.mid",
    "./dataset/clean_midi/Radiohead/Airbag.mid",
    "./dataset/clean_midi/Radiohead/Climbing Up the Walls.mid",
    "./dataset/clean_midi/Radiohead/Karma Police.mid",
    "./dataset/clean_midi/Radiohead/Subterranean Homesick Alien.mid",
    "./dataset/clean_midi/Rammstein/Stripped.mid",
    "./dataset/clean_midi/Rascel/Arrividerci Roma.mid",
    "./dataset/clean_midi/Rednex/The Ultimate Rednex Mega Mix, Part 3: Cotton Eye Joe.mid",
    "./dataset/clean_midi/Rene Carol/Rote Rosen rote Lippen roter Wein.mid",
    "./dataset/clean_midi/Rene Froger/Thunder in My Heart.mid",
    "./dataset/clean_midi/Right Said Fred/Don't Talk Just Kiss.mid",
    "./dataset/clean_midi/Rob Zombie/Demonoid Phenomenon.mid",
    "./dataset/clean_midi/Robert Palmer/I'll Be Your Baby Tonight.1.mid",
    "./dataset/clean_midi/Ross/When You Tell Me That You Love Me.mid",
    "./dataset/clean_midi/Rossi Vasco/Vivere.mid",
    "./dataset/clean_midi/Roxette/It Must Have Been Love (live studio).3.mid",
    "./dataset/clean_midi/Roxette/The Look.mid",
    "./dataset/clean_midi/Roy Orbison/Oh Pretty Woman.2.mid",
    "./dataset/clean_midi/Rush/Broon's Bane.1.mid",
    "./dataset/clean_midi/Rush/Closer to the Heart.3.mid",
    "./dataset/clean_midi/Rush/Leave That Thing Alone.mid",
    "./dataset/clean_midi/Ryan Paris/La Dolce Vita.1.mid",
    "./dataset/clean_midi/Sam Cooke/Twisting the Night Away.mid",
    "./dataset/clean_midi/Sarah McLachlan/Adia.1.mid",
    "./dataset/clean_midi/Scott McKenzie/Forrest Gump: San Francisco (Be Sure to Wear Some Flowers in Your Hair).1.mid",
    "./dataset/clean_midi/Shakatak/Night Birds.1.mid",
    "./dataset/clean_midi/Shirley Bassey/Big Spender.mid",
    "./dataset/clean_midi/Simon & Garfunkel/The Sound of Silence.2.mid",
    "./dataset/clean_midi/Sinatra/Where or When.mid",
    "./dataset/clean_midi/Stan Kenton/Here's That Rainy Day.mid",
    "./dataset/clean_midi/Steppenwolf/Born To Be Wild.3.mid",
    "./dataset/clean_midi/Stevens Shakin /You Drive Me Crazy.1.mid",
    "./dataset/clean_midi/Sting/Fields of Gold.1.mid",
    "./dataset/clean_midi/Sting/Fields of Gold.7.mid",
    "./dataset/clean_midi/Styx/Come Sail Away.4.mid",
    "./dataset/clean_midi/Supertramp/Bloody Well Right.mid",
    "./dataset/clean_midi/Supertramp/The Logical Song.7.mid",
    "./dataset/clean_midi/Survivor/Is This Love.mid",
    "./dataset/clean_midi/The Alan Parsons Project/Sirius   Eye in the Sky.mid",
    "./dataset/clean_midi/The Beach Boys/Good Vibrations.6.mid",
    "./dataset/clean_midi/The Beach Boys/I Get Around.2.mid",
    "./dataset/clean_midi/The Beatles/All You Need Is Love.3.mid",
    "./dataset/clean_midi/The Beatles/Every Little Thing.mid",
    "./dataset/clean_midi/The Beatles/Fixing a Hole.mid",
    "./dataset/clean_midi/The Beatles/Fool on the Hill.2.mid",
    "./dataset/clean_midi/The Beatles/Fool on the Hill.mid",
    "./dataset/clean_midi/The Beatles/Get Back.3.mid",
    "./dataset/clean_midi/The Beatles/Let It Be.5.mid",
    "./dataset/clean_midi/The Beatles/Octopus's Garden.mid",
    "./dataset/clean_midi/The Beatles/Paperback Writer.4.mid",
    "./dataset/clean_midi/The Beatles/Ticket to Ride.6.mid",
    "./dataset/clean_midi/The Boomtown Rats/I Don't Like Monday's.1.mid",
    "./dataset/clean_midi/The Cranberries/Animal Instinct.mid",
    "./dataset/clean_midi/The Cranberries/Promises.1.mid",
    "./dataset/clean_midi/The Doors/Riders on the Storm.1.mid",
    "./dataset/clean_midi/The Doors/The Crystal Ship.3.mid",
    "./dataset/clean_midi/The Four Seasons/Rag Doll.mid",
    "./dataset/clean_midi/The Four Seasons/Walk Like A Man.mid",
    "./dataset/clean_midi/The KLF/Justified and Ancient.1.mid",
    "./dataset/clean_midi/The Knack/My Sharona.1.mid",
    "./dataset/clean_midi/The Offspring/Gone Away.1.mid",
    "./dataset/clean_midi/The Offspring/No Hero.mid",
    "./dataset/clean_midi/The Outhere Brothers/Don't Stop (Wiggle Wiggle).1.mid",
    "./dataset/clean_midi/The Platters/Only You.5.mid",
    "./dataset/clean_midi/The Police/Every Little Thing She Does Is Magic.4.mid",
    "./dataset/clean_midi/The Police/So Lonely.1.mid",
    "./dataset/clean_midi/The Prodigy/Climbatize.mid",
    "./dataset/clean_midi/The Prodigy/Wind It Up.mid",
    "./dataset/clean_midi/The Rolling Stones/The Last Time.mid",
    "./dataset/clean_midi/The Smashing Pumpkins/Cherub Rock.mid",
    "./dataset/clean_midi/The Stylistics/I'm Stone in Love With You.mid",
    "./dataset/clean_midi/Theodorakis Mikis/Zorba's Dance.1.mid",
    "./dataset/clean_midi/Third Eye Blind/Semi-Charmed Life.1.mid",
    "./dataset/clean_midi/Third Eye Blind/Semi-Charmed Life.mid",
    "./dataset/clean_midi/Tom Jones/Delilah.5.mid",
    "./dataset/clean_midi/Tool/Flood.1.mid",
    "./dataset/clean_midi/Tool/Flood.mid",
    "./dataset/clean_midi/Toto/I Won't Hold You Back.mid",
    "./dataset/clean_midi/TURNER TINA/Notbush City Limits.1.mid",
    "./dataset/clean_midi/U2/Walk On.mid",
    "./dataset/clean_midi/U2/Where the Streets Have No Name.7.mid",
    "./dataset/clean_midi/UB40/Homely Girl.mid",
    "./dataset/clean_midi/Us3/Cantaloop.1.mid",
    "./dataset/clean_midi/Van Halen/Jump.5.mid",
    "./dataset/clean_midi/Van Halen/Me Wise Magic.mid",
    "./dataset/clean_midi/Van Halen/Right Now.1.mid",
    "./dataset/clean_midi/Vangelis/Chariots of Fire.5.mid",
    "./dataset/clean_midi/Whigfield/Saturday Night.1.mid",
    "./dataset/clean_midi/Whitney Houston/I'm Every Woman.1.mid",
    "./dataset/clean_midi/Whitney Houston/Saving All My Love For You.2.mid",
    "./dataset/clean_midi/Wonder Stevie/Happy Birthday.mid",
    "./dataset/clean_midi/Wonder Stevie/Sir Duke.1.mid",
    "./dataset/clean_midi/Yanni/Aria.1.mid",
    "./dataset/clean_midi/Yanni/Marching Season.1.mid",
    "./dataset/clean_midi/Yanni/Secret Vows.1.mid",
    "./dataset/clean_midi/Yanni/Swept Away.mid",
    "./dataset/clean_midi/Yanni/The Rain Must Fall.1.mid",
    "./dataset/clean_midi/Yazz & plastic population/The Only Way Is Up.1.mid",
    "./dataset/clean_midi/Robert Palmer/Mercy Mercy Me   I Want You.mid",
]


"""#########################################################################
listFile: return the list of midi files in the directory and subdirectories 
          of path.
input: path - the root path.
output: Dir - the list of midi files.
#########################################################################"""
def listFile(path):
    Dir = []
    for dirName, subdirList, fileList in os.walk(path):
        for name in fileList:
            midiPath = os.path.join(dirName, name)
            # Check whether the midi file is corrupted.
            if midiPath in FAILLIST:
                continue
            # Check whether the path is a midi file.
            if midiPath[-4:] == '.mid':
                Dir.append(midiPath)
    return Dir

"""#########################################################################
readMIDI: read the midi file and transfer it into piano-rolls.
input: path - the path of the midi file.
output: midi - the binary numpy array represents the piano-rolls 
        (for the convenience of my research, I remove the information of 
        the instruments).
#########################################################################"""
def readMIDI(path):
    midi = pretty_midi.PrettyMIDI(path).get_piano_roll(fs=4).T
    midi = midi > 0
    return np.asarray(midi, 'float32')


def fetchData():
    Dataset = None
    times = 1
    while 1:
        if os.path.exists(Lakh_HDF5):
            # TODO: load the .hdf5 dataset
            print("\x1b[1;34m----->> LOAD THE DATASET <<-----\x1b[0m")
            Dataset = h5py.File(Lakh_HDF5, 'r')
            break
        elif os.path.exists(Lakh_RAW):
            # pre-process the raw data and save as .hdf5
            print("Step \x1b[1;34m%d\x1b[0m: unzip the raw data." % times)
            times +=1
            # unzip the files.
            tar = tarfile.open(Lakh_RAW, "r:gz")
            tar.extractall(path='./dataset/')
            tar.close()
            Dir = listFile('./dataset/')
            print("Step \x1b[1;34m%d\x1b[0m: preprocess the raw data." % times)
            # read the raw data and save into hdf5.
            with h5py.File(Lakh_HDF5, 'w') as Dataset:
                Dataset.create_dataset('train', (1, 240, 128), maxshape=(None, 240, 128), chunks=True)
                Dataset.create_dataset('valid', (1, 240, 128), maxshape=(None, 240, 128), chunks=True)
                Dataset.create_dataset('test', (1, 240, 128), maxshape=(None, 240, 128), chunks=True)

                trainEND = 0
                validEND = 0
                testEND = 0

                L = len(Dir)
                idx = 1
                for midiPath in Dir:
                    print("\x1b[1;35m%d/%d\x1b[0m: \x1b[1;34m%s\x1b[0m" % (idx, L, midiPath))
                    idx += 1
                    midi = readMIDI(midiPath)
                    start = 0
                    while start + 240 <= midi.shape[0]:
                        rand = np.random.uniform(0, 1.0001)
                        if rand < TRAIN_RATIO:
                            # save to train.
                            Dataset['train'].resize((trainEND+1, 240, 128))
                            Dataset['train'][trainEND:trainEND+1, :, :] = np.reshape(midi[start:start+240, :],
                                                                                     [1, 240, 128])
                            trainEND += 1
                            pass
                        elif rand < Valid_RATIO:
                            # save to valid.
                            Dataset['valid'].resize((validEND + 1, 240, 128))
                            Dataset['valid'][validEND:validEND + 1, :, :] = np.reshape(midi[start:start + 240, :],
                                                                                       [1, 240, 128])
                            validEND += 1
                            pass
                        else:
                            # save to test.
                            Dataset['test'].resize((testEND + 1, 240, 128))
                            Dataset['test'][testEND:testEND + 1, :, :] = np.reshape(midi[start:start + 240, :],
                                                                                        [1, 240, 128])
                            testEND += 1
                        #
                        start += 240
                print("\x1b[1;35mFinish fetching: train(%d)/valid(%d)/test(%d)\x1b[0m"
                          % (trainEND, validEND, testEND))

        else:
            # download the raw data.
            print("Step \x1b[1;34m%d\x1b[0m: download the raw data." % times)
            times += 1
            # make a new folder to save the data.
            if not os.path.exists('./dataset'):
                os.makedirs('./dataset')
            urllib.request.urlretrieve(Lakh_URL, './dataset/clean_midi.tar.gz')

        if times > 3:
            raise(ValueError("The data fetching process is out of control!!"))

    if Dataset is None:
            raise(ValueError("The dataset is not loaded properly!!"))
    return Dataset


"""#########################################################################
MAIN UNITEST FUNCTION.
#########################################################################"""
if __name__ == '__main__':
    fetchData()