#!/usr/bin/env python3
# NIFTY 50 SENTIMENT ANALYSIS - EXECUTION SCRIPT
# ==============================================
# Execute this script to run the complete Nifty 50 sentiment analysis
# Combines FII/DII flows + Technical Analysis with maximum reliability

import asyncio
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main execution function"""
    print("NIFTY 50 SENTIMENT ANALYSIS POC")
    print("=" * 50)
    print("Components: FII/DII Flows (35%) + Technical Analysis (30%)")
    print("=" * 50)
    
    try:
        # Import the integrated analyzer
        from nifty_integrated_analyzer import test_integrated_system
        
        # Run the analysis
        print("⚡ Starting integrated analysis...")
        results = asyncio.run(test_integrated_system())
        
        if results and 'error' not in results:
            print("\n SUCCESS! Analysis completed successfully")
            print(" Check these files for detailed results:")
            print("   • nifty50_master_sentiment_analysis.json (Complete analysis)")
            print("   • nifty50_sentiment_summary.json (Quick summary)")
            print("   • fii_dii_flows_analysis.json (FII/DII details)")
            print("   • nifty_technical_analysis.json (Technical details)")
            
            # Print final summary
            sentiment = results['composite_results']['overall_sentiment']
            confidence = results['composite_results']['overall_confidence']
            interpretation = results['composite_results']['interpretation']
            
            print(f"\n FINAL RESULTS:")
            print(f"   Sentiment Score: {sentiment:+.3f}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Interpretation: {interpretation}")
            print(f"   Reliability: {'HIGH' if confidence > 0.75 else 'MODERATE' if confidence > 0.6 else 'LOW'}")
            
        else:
            print("\n❌ Analysis failed - check error messages above")
            return 1
            
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n🔧 Required files:")
        print("   • fii_dii_flows_analyzer.py")
        print("   • nifty_technical_analyzer.py") 
        print("   • nifty_integrated_analyzer.py")
        print("\nMake sure all files are in the same directory!")
        return 1
        
    except Exception as e:
        print(f"\n❌ Execution Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())