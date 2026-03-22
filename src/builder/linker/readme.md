# GCompiler Linker


## Goals

The linker is for linking all operators as a whole graph to run in aipubin, the builder(without linker) will just concatenate all text(bin) file as one file. But which is the most often case is that many operators would be used at least two times.

So the linker will reduce the text file size, it will use call to call the specific op, not just concatenate.

For further, the linker may to be as part of runtime for support absolute address jump/call.

## Desgin 

The basic element lib in this design is a file. So, by now the one file one can has one op(more will be support later).

we will cache all files in a map, the once which graph use this op, we just call it once.

the new text file layout is follow:   
||
|:--------------:|
| init_context |
| call main    |
|--------------|
| op lib       |
| op lib       |
| op lib       |
| ...          |
|--------------|
|-----main----|
| call first op     |
| call second op    |
| ...            |
| exit               |



So before `exit`, all call op will be generate by builder/linker, and linker it to a write offset;