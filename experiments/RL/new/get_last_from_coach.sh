last=$(ssh -p4242 apirrone@s-nguyen.net "cd /home/apirrone/MISC/mini_BDX/experiments/RL/new/sac/ ; ls -lt | sed -n '2 p' | grep -o '[SAC]*_[0123456789]*.zip'")

scp -p4242 apirrone@s-nguyen.net:/home/apirrone/MISC/mini_BDX/experiments/RL/new/sac/$last ./sac/