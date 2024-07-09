last=$(ssh apirrone@192.168.1.221 "cd /home/apirrone/MISC/mini_BDX/experiments/RL/new/sac/ ; ls -lt | sed -n '2 p' | grep -o '[SAC]*_[0123456789]*.zip'")

scp apirrone@192.168.1.221:/home/apirrone/MISC/mini_BDX/experiments/RL/new/sac/$last ./sac/