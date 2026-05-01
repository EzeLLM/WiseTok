#!/usr/bin/env python3

import bcrypt

passwd = b'omegamj641'
passwd1 = b'omegamj641'

salt = bcrypt.gensalt()
hashed = bcrypt.hashpw(passwd, salt)
print(len(hashed))
print(passwd)
if bcrypt.checkpw(passwd1, hashed):
    print("match")
else:
    print("does not match")