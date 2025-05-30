// command line rerouting logging to files
// go run ./cmd/web >>/tmp/info.log 2>>/tmp/error.log
package main

import (
	"flag"
	"log"
	"net/http"
	"os"
)

// Define an application struct to hold the application-wide dependencies for
// the web application. For now we'll only include fields for the two custom
// loggers, but we'll add more to it as the build progresses.
type application struct {
	errorLog *log.Logger
	infoLog  *log.Logger
}

func main() {
	// Define a new command-line flag with the name 'addr', a default value
	// of ":4000" and some short help text what the flag controls. The value
	// of the flag will be stored in the addr variable at runtime
	addr := flag.String("addr", ":4000", "HTTP network address")

	// Importantly, we use use the flag.Parse() function to parse the
	// command-line flag. This reads in the command-line flag value and
	// assigns it to the addr variable. You need to call this *before* you use
	// the addr variable otherwise it will always contain the default value
	// of ":4000". If any errors are encoutered during parsing the application
	// will be terminated.
	flag.Parse()

	// Use log.New() to create a logger for writting information messages. This
	// takes three parameters: the destination to write the logs to (os.StdOut)
	// a string prefix for message (INFO followed by a tab), and flags to
	// indicate what additional information to include (local date and time).
	// Note that the flags are joined using the bitwise OR operator |.
	infoLog := log.New(os.Stdout, "INFO\t", log.Ldate|log.Ltime)

	// Create a logger for writting error messages in the same way, but use
	// stderr as the destination and use the log.Lshortfile flag to include
	// the relevant file name and line number.
	errorLog := log.New(os.Stderr, "ERROR\t", log.Ldate|log.Ltime|log.Lshortfile)

	// Initialize a new instance of our application struct, containing the
	// dependencies.
	app := &application{
		errorLog: errorLog,
		infoLog:  infoLog,
	}
	mux := http.NewServeMux()

	// Create a file server which serves files out of the "./ui/static"
	// directory. Note that the path given to the http.Dir function is relative
	// to the project directory root.
	fileServer := http.FileServer(http.Dir("./ui/static/"))

	// Use the mux.Handle() function to register the file server as the handler
	// for all URL paths that start with "/static/". For matching paths, we
	// strip the "/static" prefix before the request reaches the file server.
	mux.Handle("/static/", http.StripPrefix("/static", fileServer))

	// Register the other application routes as methods of the app struct
	mux.HandleFunc("/", app.home)
	mux.HandleFunc("/snippet/view", app.snippetView)
	mux.HandleFunc("/snippet/create", app.snippetCreate)

	// Initialize a new http.Server struct. We set the Addr and Handler fields
	// so that the server uses the same network address and routes as before,
	// and set the ErrorLog field so that the sever now uses the custom
	// errorLog logger in the event of any problems.
	srv := &http.Server{
		Addr:     *addr,
		ErrorLog: errorLog,
		Handler:  mux,
	}

	// Write messages using the two new loggers, instead of the standard logger.
	infoLog.Printf("Starting server on %s", *addr)
	// Call the ListenAndServe() method on our new http.Server struct.
	err := srv.ListenAndServe()
	//err := http.ListenAndServe(*addr, mux)
	errorLog.Fatal(err)
}
