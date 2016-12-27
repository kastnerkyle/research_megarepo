#!/usr/bin/expect
# Example aliases:
#alias ssh_to_place="./ssh_login_expect.ex username@place"
#!/usr/bin/expect
set timeout 20
set passwd my_password
set postflags [lrange $argv 0 end]

spawn ssh {*}$postflags
expect "password:"
send "$passwd\r";
interact