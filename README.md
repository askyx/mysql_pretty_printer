## mysql pretty printer

gdb pretty printer for mysql.


a gdb pretty printer for mysql.

### Usage

* besure you have installed mysql with debug mode.
* besure you have run `mysql_config`
```
cmake -Bbuild
cmake --build build --target install
```

### pretty printer

implement some struct for mysql
* some locks like mysql_, mysql_mutex_t, mysql_cond_t
* some basic struct like

```shell
objfile xxxx/bin/mysqld pretty-printers:
  Mysql
    Init_commands_array
    List
    MYSQL_LEX_CSTRING|MYSQL_LEX_STRING
    Plugin_array
    Prealloced_array
    SQL_I_List
    Simple_cstring|Name_string|Item_name_string
    String
    collation_unordered_map
    dd_vector
    malloc_unordered_map
    mem_root_collation_unordered_map
    mem_root_deque
    mem_root_unordered_map
    mysql_cond_t
    mysql_mutex_t
    mysql_prlock_t
    mysql_rwlock_t
```

a commnd to print Query_block tree
```shell
(gdb) my qtree this
Query_term_union: 0x742dd804d9f0 parent: 0x0 qb: 0x742dd804de80
  Query_block: 0x742dd808ae58 parent: 0x742dd804d9f0
    Query_term_union: 0x742dd8078890 parent: 0x0 qb: 0x742dd8078d20
      Query_block: 0x742dd8064558 parent: 0x742dd8078890
        Query_block: 0x742dd80660d0 parent: 0x0
      Query_block: 0x742dd8077148 parent: 0x742dd8078890
  Query_block: 0x742dd807a630 parent: 0x742dd804d9f0
    Query_block: 0x742dd8045050 parent: 0x0
      Query_block: 0x742dd80459d8 parent: 0x0
  Query_block: 0x742dd804c2b0 parent: 0x742dd804d9f0
```