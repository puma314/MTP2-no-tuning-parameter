from __future__ import print_function

NUM = 2

IP_strs = {
	0:"209.97.151.22", #46bdc10f779b81de9cfe6bf648
	1:"165.227.109.96", #dfd6cd61836c4a17f71ce2dc89
	2:"165.227.183.66", #fecd5279e6a63c9b2beb9f807e
	3:"134.209.47.152", #16428aed042d8221122fd00961
	4:"159.203.80.238", #38cde48a8eceec60d4a4e92bda
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


