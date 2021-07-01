# Shell Tools and Scripting

## about '' and ""

Strings in bash can be defined with `'` and `"` delimiters, but they are not equivalent.

Strings delimited with `'` are literal strings.

Strings delimited with `"` are not literal strings. The variable values can replace the string.

```bash
foo=bar
echo "$foo"
# prints bar
echo '$foo'
# prints $foo
```

## about the special variables

- `$0` - Name of the script
- `$1` to `$9` - Arguments to the script. `$1` is the first argument and so on.
- `$@` - All the arguments
- `$#` - Number of arguments
- `$?` - Return code of the previous command
- `$$` - Process identification number (PID) for the current script
- `!!` - Entire last command, including arguments. A common pattern is to execute a command only for it to fail due to missing permissions; you can quickly re-execute the command with sudo by doing `sudo !!`
- `$_` - Last argument from the last command. If you are in an interactive shell, you can also quickly get this value by typing `Esc` followed by `.`

