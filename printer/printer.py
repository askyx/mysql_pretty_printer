import itertools
from typing import Iterator
import string
import gdb


printer = gdb.printing.RegexpCollectionPrettyPrinter('Mysql')

gdb.Command('my', gdb.COMMAND_DATA, prefix=True)


def register_printer(name):
    def __registe(_printer):
        printer.add_printer(name, '^' + name + '$', _printer)

    return __registe


def register_x_printer(name):
    def __registe(_printer):
        printer.add_printer(name, '^' + name, _printer)

    return __registe


class Printer:
    def __init__(self, val) -> None:
        self.val = val


def cast_to(val, type):
    if str(val.const_value()) == '0x0':
        return '0x0'
    return val.cast(type).dereference()


# max print 100
def getchars(arg, quote=True, length=100):
    if length == 0:
        return '""'

    if str(arg) == '0x0':
        return '0x0'

    retval = ''
    if quote:
        retval += '\''

    i = 0
    while arg[i] != ord('\0') and i < length:
        character = int(arg[i].cast(gdb.lookup_type('char')))
        try:
            if chr(character) in string.printable:
                retval += '%c' % chr(character)
            else:
                retval += '\\x%x' % character
        except ValueError:
            retval += '\\x%x' % character
        i += 1

    if quote:
        retval += '\''

    return retval


# string debug printer
@register_printer('String')
class StringPrinter(Printer):
    def to_string(self):
        return getchars(self.val['m_ptr'], True, int(self.val['m_length']))


@register_printer('Simple_cstring|Name_string|Item_name_string')
class Simple_cstringPrinter(Printer):
    def to_string(self):
        return getchars(self.val['m_str'], True, self.val['m_length'])


@register_printer('MYSQL_LEX_CSTRING|MYSQL_LEX_STRING')
class MYSQL_LEX_CSTRINGPrinter(Printer):
    def to_string(self):
        return getchars(self.val['str'], True, self.val['length'])


def num_elements(num):
    """Return either "1 element" or "N elements" depending on the argument."""
    return '1 element' if num == 1 else '%d elements' % num


def lookup_templ_spec(templ, *args):
    """
    Lookup template specialization templ<args...>.
    """
    t = '{}<{}>'.format(templ, ', '.join([str(a) for a in args]))
    try:
        return gdb.lookup_type(t)
    except gdb.error as e:
        # Type not found, try again in versioned namespace.
        global _versioned_namespace
        if _versioned_namespace not in templ:
            t = t.replace('::', '::' + _versioned_namespace, 1)
            try:
                return gdb.lookup_type(t)
            except gdb.error:
                # If that also fails, rethrow the original exception
                pass
        raise e


class StdHashtableIterator(Iterator):
    def __init__(self, hashtable):
        self._node = hashtable['_M_before_begin']['_M_nxt']
        valtype = hashtable.type.template_argument(1)
        cached = hashtable.type.template_argument(9).template_argument(0)
        node_type = lookup_templ_spec(
            'std::__detail::_Hash_node', str(valtype), 'true' if cached else 'false'
        )
        self._node_type = node_type.pointer()

    def __iter__(self):
        return self

    def __next__(self):
        if self._node == 0:
            raise StopIteration
        elt = self._node.cast(self._node_type).dereference()
        self._node = elt['_M_nxt']
        valptr = elt['_M_storage'].address
        valptr = valptr.cast(elt.type.template_argument(0).pointer())
        return valptr.dereference()


class unordered_mapPrinter(Printer):
    """Print a std::unordered_map or tr1::unordered_map."""

    def _hashtable(self):
        return self.val['_M_h']

    def to_string(self):
        count = self._hashtable()['_M_element_count']
        return 'malloc_unordered_map with {}'.format(num_elements(count))

    @staticmethod
    def _flatten(list):
        for elt in list:
            for i in elt:
                yield i

    @staticmethod
    def _format_one(elt):
        return (elt['first'], elt['second'])

    @staticmethod
    def _format_count(i):
        return '[%d]' % i

    def children(self):
        counter = map(self._format_count, itertools.count())
        # Map over the hash table and flatten the result.
        data = self._flatten(
            map(self._format_one, StdHashtableIterator(self._hashtable()))
        )
        # Zip the two iterators together.
        return zip(counter, data)

    def display_hint(self):
        return 'map'


@register_x_printer('SQL_I_List')
class SQL_I_ListPrinter(Printer):
    """Print a SQL_I_List."""

    def __init__(self, val):
        self.val = val
        self._elttype = val.type.template_argument(0)

    def to_string(self):
        return 'SQL_I_List<{}> with {}, first: {}'.format(
            self._elttype, int(self.val['elements']), self.val['first']
        )


class mem_root_dequeiter(Iterator):
    def __init__(self, node, start, end, elem_size, for_printer=False):
        self.block = node
        self.c = start
        self.e = end
        self.eme_size = elem_size
        self._count = 0
        self.for_printer = for_printer

    def __iter__(self):
        return self

    def __next__(self):
        if self.c == self.e:
            raise StopIteration

        v = self.block[self.c / self.eme_size]['elements'][self.c % self.eme_size]

        self._count = self._count + 1
        self.c = self.c + 1

        if self.for_printer:
            return ('[%d]' % self._count, v)
        else:
            return v


def FindElementsPerBlock(ele_sizeof):
    base_number_elems = 1024 / ele_sizeof
    for block_size in (16, 1025, 1):
        if block_size >= base_number_elems:
            return block_size
    return 1024


@register_x_printer('mem_root_deque')
class mem_root_dequePrinter(Printer):
    def __init__(self, val):
        self.val = val
        self._elttype = val.type.template_argument(0)
        self.block_elements = FindElementsPerBlock(self._elttype.sizeof)

    def to_string(self):
        return 'mem_root_deque<{}> with {}'.format(
            self._elttype, int(self.val['m_end_idx']) - int(self.val['m_begin_idx'])
        )

    def children(self):
        return mem_root_dequeiter(
            self.val['m_blocks'],
            self.val['m_begin_idx'],
            self.val['m_end_idx'],
            self.block_elements,
            True,
        )

    def display_hint(self):
        return 'array'


@register_x_printer('malloc_unordered_map')
class malloc_unordered_mapPrinter(unordered_mapPrinter):
    pass


@register_x_printer('collation_unordered_map')
class collation_unordered_mapPrinter(unordered_mapPrinter):
    pass


@register_x_printer('mem_root_unordered_map')
class mem_root_unordered_mapPrinter(unordered_mapPrinter):
    pass


@register_x_printer('mem_root_collation_unordered_map')
class mem_root_collation_unordered_mapPrinter(unordered_mapPrinter):
    pass


@register_x_printer('List')
class ListPrinter(Printer):
    class _iterator(Iterator):
        def __init__(self, nodetype, list):
            self._nodetype = nodetype
            self._node = list['first']
            self._elem = list['elements']
            self._count = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._elem == self._count:
                raise StopIteration
            count = self._count
            node = self._node
            val = cast_to(node['info'], self._nodetype)
            self._count = self._count + 1
            self._node = node['next']
            return ('[%d]' % count, val)

    def children(self):
        nodetype = self.val.type.template_argument(0).pointer()
        return self._iterator(nodetype, self.val)

    def to_string(self):
        if self.val['elements'] == 0:
            return 'empty List'
        return 'List'


class Prealloced_arrayPrinterBase(Printer):
    def __init__(self, val):
        super().__init__(val)
        self.container = 'Prealloced_array'
        try:
            self.nodetype = val.type.template_argument(0)
        except RuntimeError:
            self.nodetype = None

    def to_string(self):
        if self.val['m_inline_size'] == 0:
            return 'empty {}'.format(self.container)
        elif self.val['m_inline_size'] < 0 and self.val['m_ext']['m_alloced_size'] == 0:
            return 'empty {} (inline size is negative)'.format(self.container)

        m_size = (
            self.val['m_inline_size']
            if self.val['m_inline_size'] > 0
            else self.val['m_ext']['m_alloced_size']
        )
        return '{} with {}'.format(self.container, m_size)

    def children(self):
        if self.val['m_inline_size'] > 0:
            for i in range(self.val['m_inline_size']):
                yield ('[%d]' % i, self.val['m_buff'][i].cast(self.nodetype))
        else:
            for i in range(self.val['m_ext']['m_alloced_size']):
                yield (
                    '[%d]' % i,
                    self.val['m_ext']['m_array_ptr'][i].cast(self.nodetype),
                )

    def display_hint(self):
        return 'array'


@register_x_printer('Prealloced_array')
class Prealloced_arrayPrinter(Prealloced_arrayPrinterBase):
    pass


@register_printer('Init_commands_array')
class Init_commands_arrayPrinter(Prealloced_arrayPrinterBase):
    def __init__(self, val):
        super().__init__(val)
        self.nodetype = gdb.lookup_type('char *')
        self.container = 'Init_commands_array'


@register_printer('Plugin_array')
class Plugin_arrayPrinter(Prealloced_arrayPrinterBase):
    def __init__(self, val):
        super().__init__(val)
        self.nodetype = gdb.lookup_type('plugin_ref')
        self.container = 'Plugin_array'


@register_x_printer('dd_vector')
class dd_vectorPrinter(Prealloced_arrayPrinterBase):
    def __init__(self, val):
        super().__init__(val)
        self.container = 'dd_vector'


# mutex printer
PTHREAD_MUTEX_KIND_MASK = 3
PTHREAD_MUTEX_NORMAL = 0
PTHREAD_MUTEX_RECURSIVE = 1
PTHREAD_MUTEX_ERRORCHECK = 2
PTHREAD_MUTEX_ADAPTIVE_NP = 3
PTHREAD_MUTEX_DESTROYED = -1
PTHREAD_MUTEX_UNLOCKED = 0
PTHREAD_MUTEX_LOCKED_NO_WAITERS = 1
PTHREAD_MUTEX_INCONSISTENT = 2147483647
PTHREAD_MUTEX_NOTRECOVERABLE = 2147483646
FUTEX_OWNER_DIED = 1073741824
FUTEX_WAITERS = -2147483648
FUTEX_TID_MASK = 1073741823
PTHREAD_MUTEX_ROBUST_NORMAL_NP = 16
PTHREAD_MUTEX_PRIO_INHERIT_NP = 32
PTHREAD_MUTEX_PRIO_PROTECT_NP = 64
PTHREAD_MUTEX_PSHARED_BIT = 128
PTHREAD_MUTEX_PRIO_CEILING_SHIFT = 19
PTHREAD_MUTEX_PRIO_CEILING_MASK = -524288
PTHREAD_MUTEXATTR_PROTOCOL_SHIFT = 28
PTHREAD_MUTEXATTR_PROTOCOL_MASK = 805306368
PTHREAD_MUTEXATTR_PRIO_CEILING_MASK = 16773120
PTHREAD_MUTEXATTR_FLAG_ROBUST = 1073741824
PTHREAD_MUTEXATTR_FLAG_PSHARED = -2147483648
PTHREAD_MUTEXATTR_FLAG_BITS = -251662336
PTHREAD_MUTEX_NO_ELISION_NP = 512
PTHREAD_PRIO_NONE = 0
PTHREAD_PRIO_INHERIT = 1
PTHREAD_PRIO_PROTECT = 2
PTHREAD_COND_SHARED_MASK = 1
PTHREAD_COND_CLOCK_MONOTONIC_MASK = 2
COND_CLOCK_BITS = 1
PTHREAD_COND_WREFS_SHIFT = 3
PTHREAD_RWLOCK_PREFER_READER_NP = 0
PTHREAD_RWLOCK_PREFER_WRITER_NP = 1
PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP = 2
PTHREAD_RWLOCK_WRPHASE = 1
PTHREAD_RWLOCK_WRLOCKED = 2
PTHREAD_RWLOCK_READER_SHIFT = 3
PTHREAD_PROCESS_PRIVATE = 0
PTHREAD_PROCESS_SHARED = 1


MUTEX_TYPES = {
    PTHREAD_MUTEX_NORMAL: ('Type', 'Normal'),
    PTHREAD_MUTEX_RECURSIVE: ('Type', 'Recursive'),
    PTHREAD_MUTEX_ERRORCHECK: ('Type', 'Error check'),
    PTHREAD_MUTEX_ADAPTIVE_NP: ('Type', 'Adaptive'),
}


class MutexPrinter(object):
    """Pretty printer for pthread_mutex_t."""

    def __init__(self, mutex):
        """Initialize the printer's internal data structures.

        Args:
            mutex: A gdb.value representing a pthread_mutex_t.
        """

        data = mutex['__data']
        self.lock = data['__lock']
        self.count = data['__count']
        self.owner = data['__owner']
        self.kind = data['__kind']
        self.values = []
        self.read_values()

    def to_string(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_mutex_t.
        """

        return 'pthread_mutex_t'

    def children(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_mutex_t.
        """

        return self.values

    def read_values(self):
        """Read the mutex's info and store it in self.values.

        The data contained in self.values will be returned by the Iterator
        created in self.children.
        """

        self.read_type()
        self.read_status()
        self.read_attributes()
        self.read_misc_info()

    def read_type(self):
        """Read the mutex's type."""

        mutex_type = self.kind & PTHREAD_MUTEX_KIND_MASK

        # mutex_type must be casted to int because it's a gdb.Value
        self.values.append(MUTEX_TYPES[int(mutex_type)])

    def read_status(self):
        """Read the mutex's status.

        Architectures that support lock elision might not record the mutex owner
        ID in the __owner field.  In that case, the owner will be reported as
        "Unknown".
        """

        if self.kind == PTHREAD_MUTEX_DESTROYED:
            self.values.append(('Status', 'Destroyed'))
        elif self.kind & PTHREAD_MUTEX_ROBUST_NORMAL_NP:
            self.read_status_robust()
        else:
            self.read_status_no_robust()

    def read_status_robust(self):
        """Read the status of a robust mutex.

        In glibc robust mutexes are implemented in a very different way than
        non-robust ones.  This method reads their locking status,
        whether it may have waiters, their registered owner (if any),
        whether the owner is alive or not, and the status of the state
        they're protecting.
        """

        if self.lock == PTHREAD_MUTEX_UNLOCKED:
            self.values.append(('Status', 'Not acquired'))
        else:
            if self.lock & FUTEX_WAITERS:
                self.values.append(('Status', 'Acquired, possibly with waiters'))
            else:
                self.values.append(('Status', 'Acquired, possibly with no waiters'))

            if self.lock & FUTEX_OWNER_DIED:
                self.values.append(('Owner ID', '%d (dead)' % self.owner))
            else:
                self.values.append(('Owner ID', self.lock & FUTEX_TID_MASK))

        if self.owner == PTHREAD_MUTEX_INCONSISTENT:
            self.values.append(('State protected by this mutex', 'Inconsistent'))
        elif self.owner == PTHREAD_MUTEX_NOTRECOVERABLE:
            self.values.append(('State protected by this mutex', 'Not recoverable'))

    def read_status_no_robust(self):
        """Read the status of a non-robust mutex.

        Read info on whether the mutex is acquired, if it may have waiters
        and its owner (if any).
        """

        lock_value = self.lock

        if self.kind & PTHREAD_MUTEX_PRIO_PROTECT_NP:
            lock_value &= ~(PTHREAD_MUTEX_PRIO_CEILING_MASK)

        if lock_value == PTHREAD_MUTEX_UNLOCKED:
            self.values.append(('Status', 'Not acquired'))
        else:
            if self.kind & PTHREAD_MUTEX_PRIO_INHERIT_NP:
                waiters = self.lock & FUTEX_WAITERS
                owner = self.lock & FUTEX_TID_MASK
            else:
                # Mutex protocol is PP or none
                waiters = self.lock != PTHREAD_MUTEX_LOCKED_NO_WAITERS
                owner = self.owner

            if waiters:
                self.values.append(('Status', 'Acquired, possibly with waiters'))
            else:
                self.values.append(('Status', 'Acquired, possibly with no waiters'))

            if self.owner != 0:
                self.values.append(('Owner ID', owner))
            else:
                # Owner isn't recorded, probably because lock elision
                # is enabled.
                self.values.append(('Owner ID', 'Unknown'))

    def read_attributes(self):
        """Read the mutex's attributes."""

        if self.kind != PTHREAD_MUTEX_DESTROYED:
            if self.kind & PTHREAD_MUTEX_ROBUST_NORMAL_NP:
                self.values.append(('Robust', 'Yes'))
            else:
                self.values.append(('Robust', 'No'))

            # In glibc, robust mutexes always have their pshared flag set to
            # 'shared' regardless of what the pshared flag of their
            # mutexattr was.  Therefore a robust mutex will act as shared
            # even if it was initialized with a 'private' mutexattr.
            if self.kind & PTHREAD_MUTEX_PSHARED_BIT:
                self.values.append(('Shared', 'Yes'))
            else:
                self.values.append(('Shared', 'No'))

            if self.kind & PTHREAD_MUTEX_PRIO_INHERIT_NP:
                self.values.append(('Protocol', 'Priority inherit'))
            elif self.kind & PTHREAD_MUTEX_PRIO_PROTECT_NP:
                prio_ceiling = (
                    self.lock & PTHREAD_MUTEX_PRIO_CEILING_MASK
                ) >> PTHREAD_MUTEX_PRIO_CEILING_SHIFT

                self.values.append(('Protocol', 'Priority protect'))
                self.values.append(('Priority ceiling', prio_ceiling))
            else:
                # PTHREAD_PRIO_NONE
                self.values.append(('Protocol', 'None'))

    def read_misc_info(self):
        """Read miscellaneous info on the mutex.

        For now this reads the number of times a recursive mutex was acquired
        by the same thread.
        """

        mutex_type = self.kind & PTHREAD_MUTEX_KIND_MASK

        if mutex_type == PTHREAD_MUTEX_RECURSIVE and self.count > 1:
            self.values.append(('Times acquired by the owner', self.count))


class MutexAttributesPrinter(object):
    """Pretty printer for pthread_mutexattr_t.

    In the NPTL this is a type that's always casted to struct pthread_mutexattr
    which has a single 'mutexkind' field containing the actual attributes.
    """

    def __init__(self, mutexattr):
        """Initialize the printer's internal data structures.

        Args:
            mutexattr: A gdb.value representing a pthread_mutexattr_t.
        """

        self.values = []

        try:
            mutexattr_struct = gdb.lookup_type('struct pthread_mutexattr')
            self.mutexattr = mutexattr.cast(mutexattr_struct)['mutexkind']
            self.read_values()
        except gdb.error:
            # libpthread doesn't have debug symbols, thus we can't find the
            # real struct type.  Just print the union members.
            self.values.append(('__size', mutexattr['__size']))
            self.values.append(('__align', mutexattr['__align']))

    def to_string(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_mutexattr_t.
        """

        return 'pthread_mutexattr_t'

    def children(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_mutexattr_t.
        """

        return self.values

    def read_values(self):
        """Read the mutexattr's info and store it in self.values.

        The data contained in self.values will be returned by the Iterator
        created in self.children.
        """

        mutexattr_type = (
            self.mutexattr & ~PTHREAD_MUTEXATTR_FLAG_BITS & ~PTHREAD_MUTEX_NO_ELISION_NP
        )

        # mutexattr_type must be casted to int because it's a gdb.Value
        self.values.append(MUTEX_TYPES[int(mutexattr_type)])

        if self.mutexattr & PTHREAD_MUTEXATTR_FLAG_ROBUST:
            self.values.append(('Robust', 'Yes'))
        else:
            self.values.append(('Robust', 'No'))

        if self.mutexattr & PTHREAD_MUTEXATTR_FLAG_PSHARED:
            self.values.append(('Shared', 'Yes'))
        else:
            self.values.append(('Shared', 'No'))

        protocol = (
            self.mutexattr & PTHREAD_MUTEXATTR_PROTOCOL_MASK
        ) >> PTHREAD_MUTEXATTR_PROTOCOL_SHIFT

        if protocol == PTHREAD_PRIO_NONE:
            self.values.append(('Protocol', 'None'))
        elif protocol == PTHREAD_PRIO_INHERIT:
            self.values.append(('Protocol', 'Priority inherit'))
        elif protocol == PTHREAD_PRIO_PROTECT:
            self.values.append(('Protocol', 'Priority protect'))


class ConditionVariablePrinter(object):
    """Pretty printer for pthread_cond_t."""

    def __init__(self, cond):
        """Initialize the printer's internal data structures.

        Args:
            cond: A gdb.value representing a pthread_cond_t.
        """

        data = cond['__data']
        self.wrefs = data['__wrefs']
        self.values = []

        self.read_values()

    def to_string(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_cond_t.
        """

        return 'pthread_cond_t'

    def children(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_cond_t.
        """

        return self.values

    def read_values(self):
        """Read the condvar's info and store it in self.values.

        The data contained in self.values will be returned by the Iterator
        created in self.children.
        """

        self.read_status()
        self.read_attributes()

    def read_status(self):
        """Read the status of the condvar.

        This method reads whether the condvar is destroyed and how many threads
        are waiting for it.
        """

        self.values.append(
            (
                'Threads known to still execute a wait function',
                self.wrefs >> PTHREAD_COND_WREFS_SHIFT,
            )
        )

    def read_attributes(self):
        """Read the condvar's attributes."""

        if (self.wrefs & PTHREAD_COND_CLOCK_MONOTONIC_MASK) != 0:
            self.values.append(('Clock ID', 'CLOCK_MONOTONIC'))
        else:
            self.values.append(('Clock ID', 'CLOCK_REALTIME'))

        if (self.wrefs & PTHREAD_COND_SHARED_MASK) != 0:
            self.values.append(('Shared', 'Yes'))
        else:
            self.values.append(('Shared', 'No'))


class ConditionVariableAttributesPrinter(object):
    """Pretty printer for pthread_condattr_t.

    In the NPTL this is a type that's always casted to struct pthread_condattr,
    which has a single 'value' field containing the actual attributes.
    """

    def __init__(self, condattr):
        """Initialize the printer's internal data structures.

        Args:
            condattr: A gdb.value representing a pthread_condattr_t.
        """

        self.values = []

        try:
            condattr_struct = gdb.lookup_type('struct pthread_condattr')
            self.condattr = condattr.cast(condattr_struct)['value']
            self.read_values()
        except gdb.error:
            # libpthread doesn't have debug symbols, thus we can't find the
            # real struct type.  Just print the union members.
            self.values.append(('__size', condattr['__size']))
            self.values.append(('__align', condattr['__align']))

    def to_string(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_condattr_t.
        """

        return 'pthread_condattr_t'

    def children(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_condattr_t.
        """

        return self.values

    def read_values(self):
        """Read the condattr's info and store it in self.values.

        The data contained in self.values will be returned by the Iterator
        created in self.children.
        """

        clock_id = (self.condattr >> 1) & ((1 << COND_CLOCK_BITS) - 1)

        if clock_id != 0:
            self.values.append(('Clock ID', 'CLOCK_MONOTONIC'))
        else:
            self.values.append(('Clock ID', 'CLOCK_REALTIME'))

        if self.condattr & 1:
            self.values.append(('Shared', 'Yes'))
        else:
            self.values.append(('Shared', 'No'))


class RWLockPrinter(object):
    """Pretty printer for pthread_rwlock_t."""

    def __init__(self, rwlock):
        """Initialize the printer's internal data structures.

        Args:
            rwlock: A gdb.value representing a pthread_rwlock_t.
        """

        data = rwlock['__data']
        self.readers = data['__readers']
        self.cur_writer = data['__cur_writer']
        self.shared = data['__shared']
        self.flags = data['__flags']
        self.values = []
        self.read_values()

    def to_string(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_rwlock_t.
        """

        return 'pthread_rwlock_t'

    def children(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_rwlock_t.
        """

        return self.values

    def read_values(self):
        """Read the rwlock's info and store it in self.values.

        The data contained in self.values will be returned by the Iterator
        created in self.children.
        """

        self.read_status()
        self.read_attributes()

    def read_status(self):
        """Read the status of the rwlock."""

        if self.readers & PTHREAD_RWLOCK_WRPHASE:
            if self.readers & PTHREAD_RWLOCK_WRLOCKED:
                self.values.append(('Status', 'Acquired (Write)'))
                self.values.append(('Writer ID', self.cur_writer))
            else:
                self.values.append(('Status', 'Not acquired'))
        else:
            r = self.readers >> PTHREAD_RWLOCK_READER_SHIFT
            if r > 0:
                self.values.append(('Status', 'Acquired (Read)'))
                self.values.append(('Readers', r))
            else:
                self.values.append(('Status', 'Not acquired'))

    def read_attributes(self):
        """Read the attributes of the rwlock."""

        if self.shared:
            self.values.append(('Shared', 'Yes'))
        else:
            self.values.append(('Shared', 'No'))

        if self.flags == PTHREAD_RWLOCK_PREFER_READER_NP:
            self.values.append(('Prefers', 'Readers'))
        elif self.flags == PTHREAD_RWLOCK_PREFER_WRITER_NP:
            self.values.append(('Prefers', 'Writers'))
        else:
            self.values.append(('Prefers', 'Writers no recursive readers'))


class RWLockAttributesPrinter(object):
    """Pretty printer for pthread_rwlockattr_t.

    In the NPTL this is a type that's always casted to
    struct pthread_rwlockattr, which has two fields ('lockkind' and 'pshared')
    containing the actual attributes.
    """

    def __init__(self, rwlockattr):
        """Initialize the printer's internal data structures.

        Args:
            rwlockattr: A gdb.value representing a pthread_rwlockattr_t.
        """

        self.values = []

        try:
            rwlockattr_struct = gdb.lookup_type('struct pthread_rwlockattr')
            self.rwlockattr = rwlockattr.cast(rwlockattr_struct)
            self.read_values()
        except gdb.error:
            # libpthread doesn't have debug symbols, thus we can't find the
            # real struct type.  Just print the union members.
            self.values.append(('__size', rwlockattr['__size']))
            self.values.append(('__align', rwlockattr['__align']))

    def to_string(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_rwlockattr_t.
        """

        return 'pthread_rwlockattr_t'

    def children(self):
        """gdb API function.

        This is called from gdb when we try to print a pthread_rwlockattr_t.
        """

        return self.values

    def read_values(self):
        """Read the rwlockattr's info and store it in self.values.

        The data contained in self.values will be returned by the Iterator
        created in self.children.
        """

        rwlock_type = self.rwlockattr['lockkind']
        shared = self.rwlockattr['pshared']

        if shared == PTHREAD_PROCESS_SHARED:
            self.values.append(('Shared', 'Yes'))
        else:
            # PTHREAD_PROCESS_PRIVATE
            self.values.append(('Shared', 'No'))

        if rwlock_type == PTHREAD_RWLOCK_PREFER_READER_NP:
            self.values.append(('Prefers', 'Readers'))
        elif rwlock_type == PTHREAD_RWLOCK_PREFER_WRITER_NP:
            self.values.append(('Prefers', 'Writers'))
        else:
            self.values.append(('Prefers', 'Writers no recursive readers'))


@register_printer('mysql_mutex_t')
class mysql_mutex_tPrinter(MutexPrinter):
    def __init__(self, val):
        mutex = val['m_mutex']['m_u']['m_safe_ptr']['mutex']
        super().__init__(mutex)
        m_file = (
            str(val['m_mutex']['m_u']['m_safe_ptr']['file']).strip('"').split('/')[-1]
        )
        m_line = val['m_mutex']['m_u']['m_safe_ptr']['line']
        self.values.append(('Location', '{}:{}'.format(m_file, m_line)))

    def to_string(self):
        return 'mysql_mutex_t'


@register_printer('mysql_cond_t')
class mysql_cond_tPrinter(ConditionVariablePrinter):
    def __init__(self, val):
        cond = val['m_cond']
        super().__init__(cond)

    def to_string(self):
        return 'mysql_cond_t'


@register_printer('mysql_rwlock_t')
class mysql_rwlock_tPrinter(RWLockPrinter):
    def __init__(self, val):
        rwlock = val['m_rwlock']
        super().__init__(rwlock)

    def to_string(self):
        return 'mysql_rwlock_t'


@register_printer('mysql_prlock_t')
class mysql_prlock_tPrinter(Printer):
    def __init__(self, val):
        m_prlock = val['m_prlock']
        x = MutexPrinter(m_prlock['lock'])
        y = ConditionVariablePrinter(m_prlock['no_active_readers'])
        self.values = x.values
        self.values.extend(y.values)
        self.values.append(('active_readers', m_prlock['active_readers']))
        self.values.append(
            ('writers_waiting_readers', m_prlock['writers_waiting_readers'])
        )
        self.values.append(('active_writer', m_prlock['active_writer']))

    def to_string(self):
        return 'mysql_prlock_t'

    def children(self):
        return self.values


class MyTree(gdb.Command):
    def __init__(self):
        super(self.__class__, self).__init__("my qtree", gdb.COMMAND_OBSCURE)

    def invoke(self, arg, from_tty):
        args = gdb.string_to_argv(arg)
        if len(args) == 0:
            print('Usage: my qtree <item>')
            return

        item = gdb.parse_and_eval(args[0])
        self.walk(item, 0)

    def walk(self, item, level):
        item = item.cast(item.dynamic_type)

        str = "{}{}: {} ".format(
            level * "  ", item.dynamic_type.target().name, self.qbPrintThis(item)
        )

        if item.dynamic_type.target().name == 'Query_block':
            ts = item['leaf_tables']
            trs = ''
            while ts:
                trs += "{} ".format(getchars(ts['alias']))
                ts = ts['next_leaf']

            str += 'master: {} '.format(item['master'].format_string())
            str += "tables: {} ".format(trs)
            str += "select_number: {} ".format(int(item['select_number']))
            print(str)
            if item['slave']:
                subquery = item['slave']
                while subquery:
                    self.walk(subquery['m_query_term'], level + 1)
                    subquery = subquery['next']
        else:
            try:
                str += self.qbPrint(item.dereference()['m_block'])
                print(str)
            except:
                pass

        try:
            children = item.dereference()['m_children']
            for b in mem_root_dequeiter(
                children['m_blocks'],
                children['m_begin_idx'],
                children['m_end_idx'],
                FindElementsPerBlock(children.type.template_argument(0).sizeof),
            ):
                self.walk(b, level + 1)
        except:
            pass

    def qbPrintThis(self, b):
        return "{} parent: {}".format(b.format_string(), b['m_parent'].format_string())

    def qbPrint(self, b):
        str = "qb: {}".format(b.format_string())

        # TODO: ORDER, LIMIT and OFFSET
        return str


def register_mysql_printers(obj):
    gdb.printing.register_pretty_printer(obj, printer, True)


MyTree()
