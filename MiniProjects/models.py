# -*- coding: utf-8 -*-


class TaskGroup():
    '''
    This is a collection of Tasks, in case that we want to relate multiple tasks to form a more complex project
    '''

    def __init__(self):
        pass


class CommonFields():
    '''
    Fields that are common between Tasks and goals
    '''

    def __init__(self):
        self.start_date
        self.due_date
        self.end_date
        self.last_changed
        self.status
        self.owner
        self.title
        self.description
        self.material


class Task(CommonFields):
    '''
    This is the class that represents every work that has to be done by the office
    '''

    def __init__(self):
        super().__init__()
        self.status
        self.end_result
        self.end_result_pct


class Goal(CommonFields):
    '''
    This is the class where a goal of a task is stored. Each Task can have multiple goals
    '''

    def __init__(self):
        super().__init__()
        self.status
        self.result
        self.result_pct


class Event():
    '''
    Things that occur and may involve a Task or Goal
    '''

    def __init__(self):
        self.date_occured
        self.title
        self.description



class Action():
    '''
    Something that is done and affects a task or goal
    '''

    def __init__(self):
        self.date_taken
        self.title
        self.description
        self.person_acted


class Obligation():
    '''
    Something that needs to be done
    '''

    def __init__(self):
        self.date_due
        self.title
        self.description
        self.person_obliged


class Person():
    '''
    A person with many roles
    '''

    def __init__(self):
        self.first_name
        self.last_name
        self.phones
        self.emails
        self.date_created
        self.date_changed


class Structure():
    '''
    An institutional structure, company, organization
    '''

    def __init__(self):
        self.title
        self.type
        self.date_created
        self.date_changed


class Material():
    '''
    Class representing related material like docs, images etc
    '''

    def __init__(self):
        self.title
        self.type
        self.tags
        self.uri
