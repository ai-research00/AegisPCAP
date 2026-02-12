#!/bin/bash

# AegisPCAP Documentation Cleanup Script
# Removes redundant phase-specific documentation before GitHub push

echo "🧹 Starting AegisPCAP documentation cleanup..."

# Count files before cleanup
BEFORE=$(find . -maxdepth 1 -name "*.md" | wc -l)

# Remove phase-specific session summaries
rm -f PHASE_5_SESSION_SUMMARY.md
rm -f PHASE_6_1_SESSION_SUMMARY.md
rm -f PHASE_6_2_SESSION_SUMMARY.md
rm -f PHASE_6_2_STRATEGIC_THINKING.md
rm -f PHASE_6_2a_COMPLETION.md
rm -f PHASE_6_2a_INSTALLATION.md
rm -f PHASE_6_2b_COMPLETION.md
rm -f PHASE_6_2b_SESSION_SUMMARY.md
rm -f PHASE_6_3_COMPLETION.md
rm -f PHASE_6_4_COMPLETION.md
rm -f PHASE_6_4_SESSION_SUMMARY.md
rm -f PHASE_7_SESSION_1_SUMMARY.md
rm -f PHASE_7_SESSION_FINAL_SUMMARY.md
rm -f PHASE_8_SESSION_SUMMARY.md
rm -f PHASE_9_PROGRESS.md
rm -f PHASE_9_STATUS.md
rm -f PHASE_9_STEP2_COMPLETE.md
rm -f PHASE_9_STEP3_COMPLETE.md
rm -f PHASE_9_STEP4_COMPLETE.md
rm -f PHASE_10_SESSION_EXECUTION_REPORT.md
rm -f PHASE_10_DELIVERABLES_CHECKLIST.md
rm -f PHASE_11_EXECUTION_SUMMARY.md
rm -f PHASE_11_FINAL_SUMMARY.md
rm -f PHASE_12_SESSION_SUMMARY.md
rm -f PHASE_13_COMPLETION_REPORT.md
rm -f PHASE_14_FINAL_SESSION_REPORT.md
rm -f PHASE_14_DELIVERABLES_COMPLETE.md
rm -f PHASE_14_DOCUMENTATION_INDEX.md

# Remove duplicate files
rm -f "README (copy 1).md"
rm -f "TODO (copy 1).md"
rm -f "agent-instructions (copy 1).md"
rm -f "dev-mcp (copy 1).json"
rm -f "project-structure (copy 1).md"

# Remove session summaries
rm -f SESSION_SUMMARY.md
rm -f SESSION_DELIVERABLES_COMPLETE.md
rm -f SESSION_REPORT_6_4.md
rm -f SESSION_SUMMARY_PHASE_8_9.md
rm -f DAILY_SUMMARY_FEB5.md

# Remove analysis files
rm -f ANALYSIS_COMPLETE.md
rm -f COMPREHENSIVE_PROJECT_ANALYSIS.md
rm -f CURRENT_STATUS_ANALYSIS.md
rm -f IMPLEMENTATION_STATUS.md
rm -f FILE_INDEX.md
rm -f FINAL_SESSION_CHECKLIST.md
rm -f FINAL_SESSION_REPORT.md
rm -f LOCAL_INTEGRATION_TEST_REPORT.md
rm -f SYSTEM_VERIFICATION_REPORT.md

# Remove redundant quick references
rm -f QUICK_SUMMARY.md
rm -f QUICK_REFERENCE.md
rm -f QUICK_START_PHASE_9.md

# Remove phase kickoffs
rm -f PHASE_6_1_DASHBOARD_API.md
rm -f PHASE_6_2_SETUP_GUIDE.md
rm -f PHASE_7_KICKOFF.md
rm -f PHASE_7_QUICK_REF.md
rm -f PHASE_7_QUICK_REFERENCE.md
rm -f PHASE_8_KICKOFF.md
rm -f PHASE_8_OVERVIEW.md
rm -f PHASE_9_KICKOFF.md
rm -f PHASE_9_EXECUTION_PLAN.md
rm -f PHASE_10_KICKOFF.md
rm -f PHASE_10_START_HERE.md
rm -f PHASE_10_QUICK_REFERENCE.md
rm -f PHASE_10_COMPREHENSIVE_IMPLEMENTATION.md
rm -f PHASE_10_CURRENT_STATUS.md
rm -f PHASE_11_KICKOFF.md
rm -f PHASE_11_QUICK_START.md
rm -f PHASE_11_IMPLEMENTATION_GUIDE.md
rm -f PHASE_11_IMPLEMENTATION_ROADMAP.md
rm -f PHASE_11_MONITORING_HANDBOOK.md
rm -f PHASE_12_KICKOFF.md
rm -f PHASE_12_QUICK_REFERENCE.md
rm -f PHASE_13_KICKOFF.md
rm -f PHASE_13_QUICK_REFERENCE.md
rm -f PHASE_14_KICKOFF.md
rm -f PHASE_14_QUICK_REFERENCE.md

# Remove completion reports
rm -f PHASE-6-COMPLETION-REPORT.md
rm -f PHASE_5_COMPLETION.md
rm -f PHASE_7_COMPLETION.md
rm -f PHASE_7_COMPLETION_REPORT.md
rm -f PHASE_8_COMPLETION_REPORT.md
rm -f PHASE_9_COMPLETION.md
rm -f PHASE_9_FINAL_COMPLETION.md
rm -f PHASE_9_FINAL_SUMMARY.md
rm -f PHASE_10_11_SESSION_SUMMARY.md
rm -f PHASE_11_COMPLETION_REPORT.md
rm -f PHASE_12_COMPLETION_REPORT.md
rm -f PHASE_13_IMPLEMENTATION.md
rm -f PHASE_14_COMPLETION_REPORT.md
rm -f PHASE_14_FINAL_SUMMARY.md

# Remove misc files
rm -f DELIVERABLES_PHASE_8_9.md
rm -f DOCUMENTATION_INDEX.md
rm -f EXECUTIVE_SUMMARY.md
rm -f NEXT_STEPS.md
rm -f STATUS_ROADMAP_NEXT.md
rm -f STATUS_PHASE_7_FINAL.txt
rm -f PHASE_10_TASKS_SUMMARY.txt
rm -f info-misc.md
rm -f status-summary.md
rm -f START_HERE.md
rm -f START_HERE_PHASE_9.md
rm -f START_PHASE_7_HERE.md
rm -f PROJECT_STATUS_PHASE_7.md
rm -f PROJECT_STATUS_PHASE_9.md
rm -f PHASE_7_DOCUMENTATION_INDEX.md
rm -f PHASE_7_NEXT_ACTIONS.md
rm -f PHASE_4_CERTIFICATE.md
rm -f PHASE_9_QUICK_REFERENCE.md
rm -f PHASE_9_STEP4_MONITORING.md
rm -f PHASE_9_DEPLOYMENT_GUIDE.md
rm -f PHASE_7_READY_FOR_TESTING.md

# Count files after cleanup
AFTER=$(find . -maxdepth 1 -name "*.md" | wc -l)
REMOVED=$((BEFORE - AFTER))

echo "✅ Cleanup complete!"
echo "📊 Files removed: $REMOVED"
echo "📄 Remaining markdown files: $AFTER"
echo ""
echo "Essential files kept:"
echo "  ✓ README.md"
echo "  ✓ LICENSE"
echo "  ✓ CONTRIBUTING.md"
echo "  ✓ CODE_OF_CONDUCT.md"
echo "  ✓ ROADMAP.md"
echo "  ✓ TODO.md"
echo "  ✓ PROJECT_STATUS.md"
echo "  ✓ PHASE_15_PROGRESS.md"
echo ""
echo "🚀 Project is ready for GitHub push!"
