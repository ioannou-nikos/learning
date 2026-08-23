var items = [
    {
        id: 'foo',
        display: 'Foo Select'
    },
    {
        id: 'bar',
        display: 'Bar Select'
    },
];
var malakas = items.map(function (i) { return i.id; }).indexOf('foo');
console.log(malakas); // 0
