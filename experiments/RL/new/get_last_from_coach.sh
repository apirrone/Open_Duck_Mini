last=$(ssh -p4242 apirrone@s-nguyen.net "cd /home/apirrone/MISC/mini_BDX/experiments/RL/new/$1/ ; ls -lt | sed -n '2 p' | grep -io '[$1]*_[0123456789]*.zip'")

scp -p4242 apirrone@s-nguyen.net:/home/apirrone/MISC/mini_BDX/experiments/RL/new/$1/$last ./$1/