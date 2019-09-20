from __future__ import print_function

NUM = 2

#umaisthebest

IP_strs = {
	0:"209.97.151.22", #46bdc10f779b81de9cfe6bf648 random SH stability 10 with 1-[0.7, 0.8, 0.9, 0.95, 0.99, 1]
	1:"165.227.109.96", #dfd6cd61836c4a17f71ce2dc89 chain SH stability 10 with 1-[0.7, 0.8, 0.9, 0.95, 0.99, 1]
	2:"165.227.183.66", #fecd5279e6a63c9b2beb9f807e grid SH stability 10 with 1-[0.7, 0.8, 0.9, 0.95, 0.99, 1] 
	3:"134.209.47.152", #16428aed042d8221122fd00961  
	4:"159.203.80.238", #38cde48a8eceec60d4a4e92bda  star, CURRENT star_small (with small d) .. already running (afternoon)
	5:"142.93.183.29", #fc8939756e4636eb7fbad42810   
}

IP_str = IP_strs[NUM]

print('''
ssh root@{IP}

adduser umaroy
usermod -aG sudo umaroy
'''.format(IP=IP_str))

# local computer:
print('''
scp ~/Documents/meng/droplet_setup.sh umaroy@{IP}:~/
scp ~/Documents/meng/matlab.zip umaroy@{IP}:~/
'''.format(IP=IP_str))

print('''
ssh umaroy@{IP}

chmod 777 droplet_setup.sh
./droplet_setup.sh

ssh -X umaroy@{IP}
./matlab/install
/home/umaroy/R2018b
'''.format(IP=IP_str))


ROC_100: 26119 seconds
ROC_50: 24600 seconds
ROC_25: 19930 seconds

 rsync --ignore-existing -v umaroy@142.93.183.29:~/MTP2-algorithm/*.pkl .

 rsync --ignore-existing -v umaroy@165.227.183.66:~/MTP2-algorithm/*.pkl .

  rsync -v umaroy@142.93.183.29:~/MTP2-algorithm/ROC_N_500_p_100_d_0.01_data.pkl .
 
