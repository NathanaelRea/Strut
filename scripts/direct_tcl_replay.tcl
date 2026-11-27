# Capture Tcl query results and recorder files without changing the sourced model.
namespace eval ::strut_replay {
    variable domain 0
    variable recorderFiles {}
    proc noteRecorder {command code result op} {
        variable recorderFiles
        if {$code != 0} {return}
        set idx [lsearch -exact $command -file]
        if {$idx >= 0} {
            set name [lindex $command [expr {$idx + 1}]]
            dict set recorderFiles $name 1
        }
    }
    proc snapshot {} {
        variable domain
        variable recorderFiles
        incr domain
        foreach name [dict keys $recorderFiles] {
            set cwd [file normalize [pwd]]
            set source [file normalize $name]
            if {[string range $source 0 [string length $cwd]] ne "$cwd/"} {
                error "recorder output is outside the replay directory: $name"
            }
            if {[file isfile $source]} {
                set relative [string range $source [expr {[string length $cwd] + 1}] end]
                set output [file join __strut_domains $domain $relative]
                file mkdir [file dirname $output]
                # Tcl's file copy preserves timestamps, which the harness uses
                # to distinguish model inputs from generated output.
                set input [open $source rb]
                set stream [open $output wb]
                try {
                    fcopy $input $stream
                } finally {
                    close $input
                    close $stream
                }
            }
        }
        set recorderFiles {}
    }
    proc afterWipe {command code result op} {
        if {$code == 0} {snapshot}
    }
    proc query {name command code result op} {
        if {$code != 0} {return}
        set stream [open "__strut_query_${name}.out" a]
        puts $stream [join $result " "]
        close $stream
    }
}
trace add execution recorder leave ::strut_replay::noteRecorder
trace add execution wipe leave ::strut_replay::afterWipe
