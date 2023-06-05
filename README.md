# tictactoe
There are two phases to tic tac toe, training and playing. The training code is found in TrainMlannPlayer and 
the playing code is found in Game. The training system only uses random and mlann players. Bootstrap a player
by starting with an mlann player playing a random player. Do this for both player 1 and player 2 then have them 
play each other. 

## TrainMlannPlayer
To build the training code run go build main.go from inside the TrainMlannPlayer folder. 

 -episodes int

        number of games to play with this pair of players (default 10000)
 
  -epsilon float
 
        epsilon is the exploration rate for NN players (default 0.01)
 
  -gamma float
 
        gamma is the discount rate on future rewards (default 0.9)
 
  -net1 string
 
        path to the serialized player 1 NN. leave it blank to create a new one
 
  -net2 string
 
        path to the serialized player 2 NN. leave it blank to create a new one
 
  -player1 string
 
        type of player to use for player 1. One of {randoplayer, mlannplayer}
 
  -player2 string
 
        type of player to use for player 2. One of {randoplayer, mlannplayer}

For example, to have a NN player play against a random player for 10,000 games you would run, 

./main -net1 {path to where the network should be saved} -player1 mlannplayer -player2 randoplayer

Note the episodes defaults to 10,000 iterations. Also, an mlannplayer that ignores what it has learned is the same as a randoplayer. You can eliminate using a randoplayer by simply increasing
the exploration rate (epsilon). One way to bootstrap, is to set epsilon to 0.10 and then have two 
new networks play each other for a large number of iterations then reduce the exploration rate and 
have them do it again. When the players mostly tie then you're likely in a good place with your players. The games should always end in a tie when both players play optimally. 

## Game
To play against your trained player you need the following arguments -netpath {the path to the saved network}
-player {which player the network should play} -games the number of games to play against the network. 