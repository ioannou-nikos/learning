type IdDisplay = {
  id: string,
  display: string
}
const items: IdDisplay[] = [
  {
    id: 'foo',
    display: 'Foo Select'
  },
  {
    id: 'bar',
    display: 'Bar Select'
  },
]

const malakas = items.map(i => i.id).indexOf('foo');
console.log(malakas); // 0