"use strict";
// CHAPTER 03 Interfaces, Classes, Inheritance and Modules.
var __classPrivateFieldSet = (this && this.__classPrivateFieldSet) || function (receiver, state, value, kind, f) {
    if (kind === "m") throw new TypeError("Private method is not writable");
    if (kind === "a" && !f) throw new TypeError("Private accessor was defined without a setter");
    if (typeof state === "function" ? receiver !== state || !f : !state.has(receiver)) throw new TypeError("Cannot write private member to an object whose class did not declare it");
    return (kind === "a" ? f.call(receiver, value) : f ? f.value = value : state.set(receiver, value)), value;
};
var _ClassES6Private_id;
let idObject = {
    id: 2,
    name: "this is a name"
};
let optionalId = {
    id: 1
};
let optionalIdName = {
    id: 2,
    name: "optional name"
};
function printNameOrValue(obj) {
    if ('id' in obj) {
        console.log(`obj.name : ${obj.name}`);
    }
    if ('descr' in obj) {
        console.log(`obj.value : ${obj.value}`);
    }
}
printNameOrValue({
    id: 1,
    name: "nameValue"
});
printNameOrValue({
    descr: "description",
    value: 2
});
function getProperty(key, value) {
    console.log(`${key} = ${value[key]}`);
}
getProperty("id", { id: 1, name: "firstName" });
getProperty("name", { id: 2, name: "secondName" });
//getProperty("telephone", {id:3, name: "thirdName"}); // Error
//Classes and the `this` keyword
class SimpleClass {
    print() {
        console.log(`SimpleClass.print() called.`);
        console.log(`SimpleClass.id = ${this.id}`);
    }
}
let mySimpleCLass = new SimpleClass();
mySimpleCLass.id = 2020;
mySimpleCLass.print();
// Implementing interfaces
class ClassA {
    print() {
        console.log(`ClassA.print() called.`);
    }
    ;
}
class ClassB {
    print() {
        console.log(`ClassB.print() called.`);
    }
    ;
}
function printClass(a) {
    a.print();
}
let classA = new ClassA();
let classB = new ClassB();
printClass(classA);
printClass(classB);
// Class constructors
class ClassWithConstructor {
    constructor(_id) {
        this.id = _id;
    }
}
let classWithConstructor = new ClassWithConstructor(10);
console.log(`classWithConstructor = 
    ${JSON.stringify(classWithConstructor)}`);
// Class modifiers
class ClassWithPublicProperty {
}
let publicAccess = new ClassWithPublicProperty();
publicAccess.id = 10;
class ClassWithPrivateProperty {
    constructor(id) {
        this.id = id;
    }
}
let privateAccess = new ClassWithPrivateProperty(10);
//privateAccess.id = 20;
// Javascript private fields Node v12 and later
class ClassES6Private {
    constructor(id) {
        _ClassES6Private_id.set(this, void 0);
        __classPrivateFieldSet(this, _ClassES6Private_id, id, "f");
    }
}
_ClassES6Private_id = new WeakMap();
let es6PrivateClass = new ClassES6Private(10);
//es6PrivateClass.#id = 20;
//Constructor parameter properties
class ClassWithCtorMods {
    constructor(id, name) {
        this.id = id;
        this.name = name;
    }
}
let myClassMod = new ClassWithCtorMods(1, "test");
console.log(`myClassMod.id = ${myClassMod.id}`);
//console.log(`myClassMod.name = ${myClassMod.name}`);
// Readonly
class ClassWithReadonly {
    constructor(_name) {
        this.name = _name;
    }
    setNameValue(_name) {
        //this.name = _name;
    }
}
// Get and Set
class ClassWithAccessors {
    constructor() {
        this._id = 0;
    }
    get id() {
        console.log(`get id property`);
        return this._id;
    }
    set id(value) {
        console.log(`set id property`);
        this._id = value;
    }
}
let classWithAccessors = new ClassWithAccessors();
classWithAccessors.id = 10;
console.log(`classWithAccessors.id = ${classWithAccessors.id}`);
// Static functions
class StaticFunction {
    static printTwo() {
        console.log(`2`);
    }
}
StaticFunction.printTwo();
// STatic properties
class StaticProperty {
    updateCount() {
        StaticProperty.count++;
    }
}
StaticProperty.count = 0;
let firstInstance = new StaticProperty();
let secondInstance = new StaticProperty();
firstInstance.updateCount();
console.log(`StaticProperty.count = ${StaticProperty.count}`);
secondInstance.updateCount();
console.log(`StaticProperty.count = ${StaticProperty.count}`);
// Namespaces
var FirstNameSpace;
(function (FirstNameSpace) {
    class NameSpaceClass {
    }
    FirstNameSpace.NameSpaceClass = NameSpaceClass;
    class NotExported {
    }
})(FirstNameSpace || (FirstNameSpace = {}));
let nameSpaceClass = new FirstNameSpace.NameSpaceClass();
class IdNameClass {
    constructor() {
        this.id = 0;
        this.name = "nameString";
    }
}
let multipleObject = {
    id: 1,
    name: "myName",
    description: "myDescription"
};
// Class inheritance
class BaseClass {
    constructor() {
        this.id = 0;
    }
}
class DerivedFromBaseClass extends BaseClass {
    constructor() {
        super(...arguments);
        this.name = "nameString";
    }
}
class MultipleInterfaces {
    constructor() {
        this.id = 0;
        this.name = "nameString";
    }
}
// The `super` function
class BaseClassWithCtor {
    constructor(id) {
        this.id = id;
    }
}
class DerivedClassWithCtor extends BaseClassWithCtor {
    constructor(id, name) {
        super(id);
        this.name = name;
    }
}
// Function Overriding
class BaseClassWithFn {
    print(text) {
        console.log(`BaseClassWithFn.print(): ${text}`);
    }
}
class DerivedClassFnOverride extends BaseClassWithFn {
    print(text) {
        console.log(`DerivedClassFnOverride.print(): ${text}`);
    }
}
let derivedClassFnOverride = new DerivedClassFnOverride();
derivedClassFnOverride.print("test");
class DerivedClassFnCallthrough extends BaseClassWithFn {
    print(text) {
        super.print(`from DerivedClassFnCallthrough : ${text}`);
    }
}
let derivedCallthrough = new DerivedClassFnCallthrough();
derivedCallthrough.print("text");
//# sourceMappingURL=chapter03.js.map